# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# remove_cell
import sys
sys.path.insert(0, '/home/jovyan/ros/')

# %load_ext autoreload
# %autoreload 2

# +
# remove_cell
import itertools as it
import logging
import operator

import altair as A
import arviz as az
from bambi import Model
import dscontrib.wbeard as wb
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import pandas as pd
import scipy.stats as st
from numba import njit
import toolz.curried as z
import seaborn as sns

from ros.utils.common import hstack, vstack, plot_wrap, drop_outliers, disable_logging
from ros.utils import bootstrap as bs, plot as plu

disable_logging(["numba", "arviz", "pymc3", "bambi", "numexpr"])  # 

str_concat = z.compose("-".join, z.map(str))
lmap = z.comp(list, map)
plt.rcParams["font.size"] = 17

p = lambda: None
p.__dict__.update(
    dict(
        zip(
            "hide_output hide_input collapse_hide collapse_show remove_cell".split(),
            range(10),
        )
    )
)


# -

# There's an interplay between sample size, effect size, and the sensitivity of an experiment to detect changes that we spend time thinking about at Mozilla. All else equal, it's usually preferable to enroll a smaller sample size so long as it's sufficient to pick up the signal of the treatment. Among other reasons, this helps reduce the likelihood of different experiments interacting with each other. But there are ways to increase the resolution of experimental analysis without having to increase the population size, and this efficient frontier is a good place to aim for.
#
# Some techniques I've tried lately are using [blocking](https://en.wikipedia.org/wiki/Blocking_(statistics)) and pre-treatment predictors as useful ways to get more precise estimates for free, without the need of a larger study population. This post simulates experimental data to demonstrate the improvement in precision that you can get with these. 

# ## The Setup
#
# The example here is a study that measures an improvement in startup times of a new feature. This is a metric we pay quite a bit of attention to in the platform area, and are obviously interested in features that can reduce startup times. The study population in this example has a distribution of startup times, but on average Windows 7 users have longer times than Windows 10 users.[1] 
#
# The basic idea with blocking is that if 2 groups in the population have significantly different outcomes, independent of the treatment variable, you can get a more precise estimate of the treatment effect by modeling these groups separately.
# Intuitively, if the 2 groups have significantly different outcomes even before the treatment is applied, this difference will contribute to a higher variance in the estimate when it comes time to measure the size of the treatment effect. The variable that determines the grouping needs to be independent of the treatment assignment, so using Windows 7 as a blocking factor would be a good choice, as our feature doesn't do anything preposterous like upgrade the OS once the client enrolls.
#
# The second idea is to use pre-treatment variables as a predictor. In this case, it involves looking at the startup time before enrollment, and seeing how much this changes on average for the treatment group once they get the feature. This works if a 
# client's pre-treatment startup time $t_{pre}$ is more informative of the post-treatment startup time $t_{post}$ than merely knowing the OS version, and it's safe to assume here that $t_{post}$ and the OS are conditionally independent given $t_{pre}$. 
#
# As with many metrics we use, the log of the startup time more closely follows the distributions we're used to. For this simulation we'll set the log of the first_paint time to follow a gamma distribution, with the mean time increased for Windows 7 users.[2] For users in the treatment group, we'll add a noisy log(.9)  (=-.105) to the distribution, which translates to roughly a 10% decrease in startup times on the linear scale.[3] After the simulation, we'll look at how much of an improvement you get with the estimates when using a randomized block design. The formulas describing the simulation are
#
#
# \begin{align}
# fp_{baseline} & \sim \mathrm{Gamma}(4 + \mathbb 1_{win7} \cdot \mu_{win}) \\
# w_{treat} & \sim \mathcal N (\mathbb 1_{treat} \cdot \mu_{treat}, \sigma_{treat}) \\
# \log(first\_paint) & = fp_{baseline} + w_{treat}
# \end{align}

# +
# hide_input
@njit
def seed(n):
    nr.seed(n)

@njit
def randn(mu, sig, size=1):
    return nr.randn(size) * sig + mu


# +
# collapse_show
WIN7_FACT = 1.2
TREAT_MU = np.log(.9)
TREAT_SD = .15


@njit
def gen_log_first_paint_pre_post(win7, treat, size=1):
    pre = nr.gamma(4 + WIN7_FACT * win7, 1, size=size)
    return np.concatenate((pre, pre + randn(TREAT_MU * treat, TREAT_SD, size=size)))


# +
# collapse_hide
n_each = 10_000
n_win_7 = {0: n_each, 1: n_each}
seed(0)


def add_columns(df):
    pre_post = pd.DataFrame(
        [
            gen_log_first_paint_pre_post(win7, treat=treat)
            # gen_pre_post(win7, win7_fact=WIN7_FACT, treat=treat, treat_fact=TREAT_FACT,)
            for win7, treat in df[["win7", "treat"]].itertuples(index=False)
        ],
        columns=["lpre", "lpost"],
    ).assign(
        pre=lambda df: np.exp(df.lpre),
        post=lambda df: np.exp(df.lpost),
    )
    df = hstack([df, pre_post])
    df = (
        df.assign(os=lambda df: df.win7.map({0: "win10", 1: "win7"}))
        .reset_index(drop=0)
        .rename(columns={"index": "id"})
    )
    df["demo"] = [
        str_concat(tup)
        for tup in df[["treat", "os"]]
        .assign(treat=lambda df: df.treat.map({1: "treat", 0: "control"}))
        .itertuples(index=False)
    ]
    return df


def create_test_pop(n_each=50):
    data_dct = [
        {"win7": win7, "treat": treat}
        for win7 in (0, 1)
        for treat in (0, 1)
        for _ in range(n_win_7[win7])
    ]
    df_ = pd.DataFrame(data_dct)
    df = df_.pipe(add_columns)
    return df


def stack_pre_post(df):
    """
    demo[graphic] is concatenated values for
    ["treat", "os"]
    demo2 is for ["treat", "os", "pre"]
    """
    dfs = (
        df.set_index(["id", "win7", "os", "treat", "demo"])
        .stack()
        .reset_index(drop=0)
        .rename(columns={"level_5": "pre", 0: "y"})
        .assign(
            demo2=lambda df: lmap(
                str_concat, df[["demo", "pre"]].itertuples(index=False)
            )
        )
        .assign(pre=lambda df: (df.pre == "pre").astype(int))
    )
    return dfs


treat_i2name = {0: "control", 1: "treat"}
df = create_test_pop().assign(branch=lambda df: df.treat.map(treat_i2name))
dfs = stack_pre_post(df.drop(["lpre", "lpost", "branch"], axis=1)).assign(
    branch=lambda df: df.treat.map(treat_i2name)
)
# -

# The data is in the following format:

# remove_cell
df.sample(frac=1, random_state=0).drop_duplicates(["win7", "treat"]).drop(
    ["demo", "win7", "lpre", "lpost", 'treat'], axis=1
).reset_index(drop=1)

# collapse_hide
dfs.sample(frac=1, random_state=0).drop_duplicates(["win7", "treat", "pre"]).drop(
    ["demo", "win7", 'treat', 'demo2'], axis=1
).reset_index(drop=1)[:6]

# and can be split into 4 groups based on the OS version (`win7` or `win10`) and study branch (`treat` or`control`). Each client (identified by an integer `id` value) has a measurement before and after enrolling into the experiment, denoted by a binary flag in the `pre` column. This pre- and post-measurement will help get more a more sensitive estimate of the treatment effect. 
#
# The following histograms show the distributions for pre- and post-enrollment measurements:

# +
# collapse_hide
axs, spi = wb.plot_utils.mk_sublots(nrows=1, ncols=2, figsize=(16, 5), sharex=False)

for pre, pdf in dfs.groupby("pre"):
    spi.n
    plt.title(["Pre-enrollment", "Post-enrollment"][pre])
    for branch, bdf in pdf.groupby("branch"):
        lbins = np.logspace(0, np.log(bdf.y.max()), 100)
        plt.hist(bdf.y, bins=lbins, density=1, alpha=0.5, label=branch)

    plt.legend()
    plt.xscale("log")

    if not pre:
        plt.ylabel("Density")
    plt.xlabel("Simulated first paint (ms)")
# -

# These are mostly overlapping, and show roughly gamma distributed times on the log scale. Notice that while most of the density is less than $10^4=10,000$ ms (~96% to be more precise), the scale extends to around $10^{19}$, indicating some serious outliers.
#
# The plot below shows the quantile ranges covering the middle 50% (thick lines) and middle 95% (thin lines) of times, with the median represented with the dot for the different slices of data.

# collapse_hide
plu.plot_groupby(dfs, 'demo2', 'y', quants=[50, 95])
plt.xscale('log')

# This plot gives a finer view of the difference between the distributions. Each pair for the control branch (the first four lines) should have identical values, but for the treatment groups, the `post` measurements should be slightly less than the `post` measurements. Note that the Windows 7 distributions are significantly longer, and that this difference in OS times is larger than the difference in treatment.

# ## Difference in means

# remove_cell
dfs_pre = dfs.query("pre == 1")
dfs_pre.groupby("branch").y.mean()


# A common way of evaluating the effect of a treatment is by looking at the difference in the mean of each group, with bootstrapping to measure the uncertainty of the statistic. (Comparing branches throughout the distribution can give a lot more insight, but comparing means keeps the illustration simpler). So a standard way to compare the difference between branches would be to just take the mean of the `post` measurements for each branch (which are representative of the times while the experiment is running), take the difference between these, and bootstrap this statistic.

# +
# collapse_hide
def bootstrap_stat_diff(
    df, gbcol, ycol, stat_fn=np.mean, n_reps=10_000, comp=operator.sub
):
    """
    `gbcol` needs to be col of 0 and 1's, designating
    the groups being compared.
    """
    control = df[ycol][df[gbcol] == 0].values
    treat = df[ycol][df[gbcol] == 1].values
    size = len(control)
    stats = []

    for _ in range(n_reps):
        ac = nr.choice(control, size=size, replace=True)
        at = nr.choice(treat, size=size, replace=True)
        #         stats.append(stat_fn(at) - stat_fn(ac))
        stats.append(comp(stat_fn(at), stat_fn(ac)))

    return np.array(stats)


def uplift(a, b):
    return a / b - 1


def plot_bs(diffs, log=False, title=None, pe=1, ax=None):
    diffs_p = plot_wrap(diffs)
    plu.plot_groupby(
        diffs_p,
        "g",
        "y",
    )
    plt.yticks([0], ["Post-enrollment" if pe else ""])
    if log:
        plt.xscale("log")
    plt.title(title)


dfs_post = dfs.query("pre == 0").assign(yl=lambda df: df.y.pipe(np.log))

# +
# collapse_hide

diffs = bootstrap_stat_diff(dfs_post, "treat", "y")
diffs_out_rm = bootstrap_stat_diff(dfs_post.pipe(drop_outliers), "treat", "y")
diffs_log = bootstrap_stat_diff(dfs_post, "treat", "yl")
diffs_gmean = bootstrap_stat_diff(dfs_post, "treat", "y", stat_fn=st.gmean)
ul_out_rm = bootstrap_stat_diff(dfs_post.pipe(drop_outliers), 'treat', 'y', comp=uplift)
ul = bootstrap_stat_diff(dfs_post, 'treat', 'y', comp=uplift)
# -

# A first thing to note is that doing this is a Bad Idea with log-normal distributed data. The following show both the bootstrapped differences in means and bootstrapped uplift (% increase in times of the treatment over the control group):

# +
# hide_input
axs, spi = wb.plot_utils.mk_sublots(nrows=2, ncols=1, figsize=(8, 8), sharex=False)

plot_bs(diffs, log=1, title='Difference in means', ax=spi.n)
plot_bs(ul * 100, log=0, title='% Increase in treatment', pe=1, ax=spi.n)
plt.tight_layout()
# -

# Instead of showing a _decrease_ in times from the treatment group, these estimates show an _increase_ in the times. And an increase on the order of 200%! This goes back to the histograms above that showed outliers way higher than the median time. These bootstrapped estimates above are essentially dominated by a handful of outliers; the top 5 are all from the treatment group:

# hide_input
dfs_post.sort_values('y', ascending=False)[:5][['branch', 'y']][::-1].reset_index(drop=1)

# When the outliers (arbitrarily chosen to be the top .01%) are removed, the estimates look a bit more sane:

# +
# hide_input
axs, spi = wb.plot_utils.mk_sublots(nrows=2, ncols=1, figsize=(8, 10))

plot_bs(diffs_out_rm, log=0, title='Difference in means', pe=0, ax=spi.n)

plot_bs(ul_out_rm, log=0, title='% Increase in treatment', pe=0, ax=spi.n)
plt.suptitle("Outliers removed");
# plt.tight_layout();
# -

# In fact, the ~10% decrease we programmed into the simulation is now showing up as a plausible value in the plot below, but these estimates also give a relatively high likelihood of there being no effect (that is, 0 is within the 50% CI).
#
# Taking the log and bootstrapping those means, however, similarly allows for a 10% decrease in startup times, but this time rules out a positive effect:

plot_bs(diffs_log, log=0, title='Difference in mean of log(startup time)', pe=0)

# Note that this is very similar to taking the geometric mean (after taking the mean of the log, you'd just need to exponentiate the result), which I often prefer as a more robust statistic on this kind of data, for exactly this reason.

# ## Linear regression as a difference in means

# +
# remove_cell
fit_kwa = dict(samples=1000, chains=4, backend='pymc3', cores=4, target_accept=.8)

data = dfs.assign(yl=lambda df: df.y.pipe(np.log))


# -

# Another way to accomplish the difference in means from above is with Bayesian linear regression. I'm using [Bambi](https://github.com/bambinos/bambi) with [pymc3](https://github.com/pymc-devs/pymc3/) for easy integration with python, and we'll be operating on the log from here on out due to the stability issues shown above. Bambi allow r-like formula syntax, and we can start off with the simplest regression: `log(y) ~ treat`, where `treat` is 1 for clients in the treatment group and 0 otherwise. When run on the post-enrollment data, the coefficient for `treat` gives a similar estimate as the bootstrapped mean does:

# collapse_show
def fit_regression(df):
    return Model(df).fit("yl ~ treat", **fit_kwa)


# remove_cell
fit = fit_regression(data.query("pre == 0").copy())

# +
# hide_input
posterior_treatment = plot_wrap(fit.posterior.treat.values.ravel())

post_plus_boot = vstack([
    plot_wrap(diffs_log).assign(g='bootstrapped mean'),
    posterior_treatment.assign(g='linear model posterior')
])

plu.plot_groupby(post_plus_boot, 'g', 'y')
yo, yi = plt.ylim()
plt.vlines([TREAT_MU, TREAT_MU], yo, yi, linestyles='dashed');


# -

# I hadn't realized this, but it makes sense in hindsight; if the value of the indicator variable is 1, then the slope (aka the coefficient) is going to be equivalent to the mean of the difference between the two groups.

# ## Regression with Blocking and baseline

# Now we get what we came here for. The model to use OS version as a blocking variable will be `log(y) ~ treat + win7`, and the formula that adds pretreatment will be `log(y_post) ~ log(y_pre) + treat + win7`. 

# +
def fit_block_regression(df):
    return Model(df.copy()).fit("log_post_startup ~ treat + win7", **fit_kwa)

def fit_block_baseline_regression(df):
    return Model(df.copy()).fit("log_post_startup ~ log_baseline_startup + treat + win7", **fit_kwa)


# -

# remove_cell
df_pre_post = df.assign(log_baseline_startup=lambda df: df.pre.pipe(np.log), log_post_startup=lambda df: df.post.pipe(np.log))
fit_block = fit_block_regression(df_pre_post)
posterior_treatment_block = plot_wrap(fit_block.posterior.treat.values.ravel())

# remove_cell
fit_block_baseline = fit_block_baseline_regression(df_pre_post)
posterior_treatment_block_bl = plot_wrap(fit_block_baseline.posterior.treat.values.ravel())

# The estimated treatment effects of all the models run so far are below. The model with blocking shows only a slight improvement; the extent of the 95% credible interval is slightly, but visibly less than those of the previous models, and the median estimate moved a tiny bit toward the true effect (shown with the dashed line).

# +
# hide_input
plt.figure(figsize=(12, 5))
post_boot_block = vstack([
    posterior_treatment.assign(g='2- linear model posterior'),
    posterior_treatment_block.assign(g='3- linear model posterior:\nblocked'),
    posterior_treatment_block_bl.assign(g='4- linear model posterior:\nblocked + baseline'),
    plot_wrap(diffs_log).assign(g='1- bootstrapped mean'),
])

plu.plot_groupby(post_boot_block, 'g', 'y', quants=[50, 95], label='_no_label_')
yo, yi = plt.ylim()
plt.vlines([TREAT_MU, TREAT_MU], yo, yi, linestyles='dashed', label='true effect')
plt.legend();
# -

# But the real difference was when the pre-treatment effects were included. The estimate with the baseline predictor honed right in on the true effect, and the credible intervals tightened to a fraction of what the raw log mean estimates would have been. In this case, the baseline measurements increased both the accuracy and precision of the estimate remarkably.

# ## Summary and Caveats
#
# This post hopefully makes clear the benefits of including the baseline measure, but a contrived example like this requires some caveats. Two major ones to be mindful of are the magnitude and consistency of the effect size across variables.  A 10% effect size is very large among the experimental results I've seen, so this example isn't representative in that respect (though this may highlight the importance of including baseline measurements!). I also added a constant 10% improvement to all clients in the treatment branch, regardless of their baseline measurement. From the actual data I've seen, though, the effect is very non-linear, partly due to a regression to the mean effect, and partly due to other factors.
#
# Nevertheless, including the baseline measurement enables more precise estimates, especially when there's an imbalance in the experiment branches. If there had been more Windows 7 users in the treatment branch through some fluke of randomization, then a naive comparison of the means could underestimate the effect size if baselines aren't taken into account.
#
# ### TLDR highlights
#
# - When working with data that appears to be log-distributed, looking at the difference in mean can give you noisy results
# - When comparing 2 groups, modeling a group using an indicator variable as a linear predictor is equivalent to modeling the difference in means of the group
# - Block variables and baseline measurements are a good way to improve the estimates of the effect of a treatment 


