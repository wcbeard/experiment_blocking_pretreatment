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
import operator

import altair as A
import dscontrib.wbeard as wb
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import pandas as pd
import scipy.stats as st
from numba import njit
import toolz.curried as z
import seaborn as sns


from ros.utils.common import hstack, vstack, plot_wrap, drop_outliers
from ros.utils import plot as plu
from ros.utils import bootstrap as bs

import dscontrib.wbeard as wb

str_concat = z.compose("-".join, z.map(str))
lmap = z.comp(list, map)
plt.rcParams["font.size"] = 17
# -

# # Build dataset

# +
# collapse_hide
WIN7_FACT = 1.2
TREAT_MU = np.log(.9)
TREAT_SD = .15

@njit
def seed(n):
    nr.seed(n)

@njit
def randn(mu, sig, size=1):
    return nr.randn(size) * sig + mu

@njit
def gen_log_first_paint_pre_post(win7, treat, size=1):
    pre = nr.gamma(4 + WIN7_FACT * win7, 1, size=size)
    return np.concatenate((pre, pre + randn(TREAT_MU * treat, TREAT_SD, size=size)))


# -

# \begin{align}
# fp_{baseline} & \sim \mathrm{Gamma}(4 + \mathbb 1_{win7} \cdot \mu_{win}) \\
# w_{treat} & \sim \mathcal N (\mathbb 1_{treat} \cdot \mu_{treat}, \sigma_{treat}) \\
# \log(first\_paint) & = fp_{baseline} + w_{treat}
# \end{align}

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
df = create_test_pop()
dfs = stack_pre_post(df.drop(["lpre", "lpost"], axis=1)).assign(
    branch=lambda df: df.treat.map(treat_i2name)
)
df.drop_duplicates("win7")
# -

dfs[:3]

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

# - control: no difference pre/post
# - for the treatment groups post is slightly larger than pre

# collapse_hide
plu.plot_groupby(dfs, 'demo2', 'y', quants=[50, 95])
plt.xscale('log')

# # Difference in means

# collapse_hide
dfs_pre = dfs.query("pre == 1")
dfs_pre.groupby("branch").y.mean()

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
    plt.yticks([0], ["Pre-enrollment" if pe else ""])
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

# +
# hide_input
axs, spi = wb.plot_utils.mk_sublots(nrows=1, ncols=2, figsize=(16, 5), sharex=False)

plot_bs(diffs, log=1, title='Difference in means', ax=spi.n)
plot_bs(ul * 100, log=0, title='% Increase in treatment', pe=0, ax=spi.n)
# -

# hide_input
dfs_post.sort_values('y', ascending=False)[:5][['branch', 'y']][::-1].reset_index(drop=1)

# collapse_hide
dfs_post.query("yl > 16")[['branch', 'y']].sort_values('y', ascending=True).reset_index(drop=1)

# +
axs, spi = wb.plot_utils.mk_sublots(nrows=1, ncols=2, figsize=(16, 5))

plot_bs(diffs_out_rm, log=0, title='Difference in means', pe=0, ax=spi.n)

plot_bs(ul_out_rm, log=0, title='% Increase in treatment', pe=0, ax=spi.n)
plt.suptitle("Outliers removed");
# -

plot_bs(diffs_log, log=0, title='Diffs log')

# +
axs, spi = wb.plot_utils.mk_sublots(nrows=3, ncols=2, figsize=(16, 15))

spi.n
plot_bs(diffs, log=1, title='Difference in means')

spi.n
plot_bs(diffs_out_rm, log=0, title='Mean diff, outlier rm', pe=0)

spi.n
plot_bs(diffs_log, log=0, title='Diffs log')

spi.n
plot_bs(np.exp(diffs_log), log=0, title='Diffs log exp', pe=0)

spi.n
plot_bs(diffs_gmean, log=0, title='Diffs gmean', pe=0)

spi.n
plot_bs(ul_out_rm, log=0, title='Uplift, outlier rm', pe=0)

plt.tight_layout()
# -

dfs_pre.assign(
    yl=lambda df: df.y.pipe(np.log)
).groupby("treatn").yl.mean()

# Top .01%

# ## As linear regression

# +
from bambi import Model
import arviz as az

data = dfs.assign(yl=lambda df: df.y.pipe(np.log))
# -

data[:3]

mod = Model(data.query("pre == 0").copy())

fit = mod.fit("yl ~ treat", samples=1000, chains=4, target_accept=.8)

posterior_treatment = plot_wrap(fit.posterior.treat.values.ravel())

post_plus_boot = vstack([
    plot_wrap(diffs_log).assign(g='bootstrapped_mean'),
    posterior_treatment.assign(g='linear model posterior')
])

plu.plot_groupby(post_plus_boot, 'g', 'y')
yo, yi = plt.ylim()
plt.vlines([TREAT_MU, TREAT_MU], yo, yi, linestyles='dashed');

# # Blocking

df.groupby(['os', 'treat']).post.median().unstack()

df_pre_post = df.assign(pre_l=lambda df: df.pre.pipe(np.log), post_l=lambda df: df.post.pipe(np.log))
mod_block = Model(df_pre_post.copy())

fit_block = mod_block.fit("post_l ~ treat + win7", samples=1000, chains=4, backend='pymc3')

posterior_treatment_block = plot_wrap(fit_block.posterior.treat.values.ravel())

fit_block_baseline = mod_block.fit("post_l ~ pre_l + treat + win7", samples=1000, chains=4)

posterior_treatment_block_bl = plot_wrap(fit_block_baseline.posterior.treat.values.ravel())

# +
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

# 'pre_l',  'win7'
az.plot_forest(fit_block, var_names=['treat', ], combined=True)

TREAT_FACT

az.summary(fit_block)

df_pre_post[:3]

df_pre_post[:3]

data[:3]

# # Plots

# +
import dscontrib.wbeard as wb
axs, spi = wb.plot_utils.mk_sublots(nrows=1, ncols=2, figsize=(16, 5))

spi.n
df = create_test_pop(5000)
plu.plot_groupby(df, 'os', 'pre')
plt.xscale('log')

spi.n
df = create_test_pop(5000)
plu.plot_groupby(df, 'os', 'pre')
plt.xscale('log')


# +
def mk_bootstrap_df(bs_dct, n_reps=1_000):
    dfs = [
        pd.DataFrame({'y': bs.draw_replicates(n_reps)}).assign(g=k)
        for k, bs in bs_dct.items()
    ]
    df = pd.concat(dfs)
    return df

means_ = {
    k: wb.bootstrap.BootstrapStat(gdf.pre, stat=np.mean)
    for k, gdf in df.groupby('os')
}
mean_draws = mk_bootstrap_df(means_, n_reps=10_000)
# -

plu.plot_groupby(mean_draws, 'g', 'y', quants=[50, 95])
plt.xscale('log')

df[:3]

# +

df = create_test_pop(5000)

plot_groupby(df.assign(os=lambda df: df.win7.map({0: 'win10', 1: 'win7'})), 'os', 'pre')

plt.xscale('log')

# +

        
plot_probs(sp)
# -



df.groupby(['win7'])[['pre', 'treat']].quantile([.025, .25, .75, .975]).unstack()
# .T.unstack()

df.groupby('win7').pre.mean()

# +

for win7, gdf in df.groupby('win7'):
    1
    
del win7
# -

win7_factor = 1.2

# +



# -

plt.plot(10 ** xs, ys)


# # Junk

# +
def gen_times(win7):
    times = st.gamma(4 + win7_factor * win7).rvs(len(gdf))
    return times

def gen_times_shape(gshape):
    times = st.gamma(gshape).rvs(len(gdf))
    return times

# +


def gamma_shape(win7):
    return 4 + win7_factor * win7
    
def add_columns(df):
    df['gshape'] = gamma_shape(df.win7)
#     df['pre'] = df.groupby('win7').win7.transform(gen_times)
    df['pre'] = df.groupby('win7').gshape.transform(gen_times_shape)
    df['treat'] = df['pre'] + nr.randn(len(df)) * .25 + 1
    return df

df = pd.DataFrame(data_dct)
df = df.pipe(add_columns)
df.drop_duplicates()
