import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

# import numpy.random as nr  # type: ignore
# import pandas as pd  # type: ignore


def intervals(n):
    """
    0 <= n <= 100
    50% -> [.25, .75]
    """
    pm = n / 2 * 0.01
    return np.round(0.5 - pm, 3), np.round(0.5 + pm, 3)


def samp_probs(df, gb, col, quants=[50, 95]):
    """
    =>
                 0.025  0.250   0.500   0.750   0.975
    win7
    0   1.291482    2.291133    3.757434    4.832630    6.746378
    1   1.893204    3.907341    5.420873    6.753467    9.801736
    """
    p1a, p1b = intervals(quants[0])
    p2a, p2b = intervals(quants[1])
    return df.groupby(gb)[col].quantile([p2a, p1a, 0.5, p1b, p2b]).unstack()


def plot_probs(pdf, reverse=False, label='ix'):
    """
    Take a df with quantile columns, iterate over rows and plot
    quantiles with different thickness:
                     0.025  0.250   0.500   0.750   0.975
    win7
    0   1.291482    2.291133    3.757434    4.832630    6.746378
    1   1.893204    3.907341    5.420873    6.753467    9.801736
    - Index will be y label.
    """
    ix_labels = []
    i_vals = []
    if reverse:
        pdf = pdf[::-1]

    for i, (ix, *row) in enumerate(pdf.itertuples(index=True, name=None)):
        # print(row)
        ix_labels.append(ix)

        i_vals.append(i)
        q1a, q2a, mid, q2b, q1b = row
        plt.plot([q1a, q1b], [i, i], "k", linewidth=1, label=label)
        plt.plot([q2a, q2b], [i, i], "k", linewidth=3)
        plt.plot(mid, i, "ok")
    plt.yticks(i_vals, ix_labels)


def plot_groupby(df, gb, col, quants=[50, 95], reverse=True, label='ix'):
    pdf = samp_probs(df, gb, col, quants=quants)
    return plot_probs(pdf, reverse=reverse, label=label)
