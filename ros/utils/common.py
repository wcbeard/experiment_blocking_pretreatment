from functools import partial

import pandas as pd  # type: ignore

vstack = partial(pd.concat, ignore_index=True)
hstack = partial(pd.concat, ignore_index=False, axis=1)
plot_wrap = lambda y: pd.DataFrame(dict(y=y, g=1))


def drop_outliers(df, ycol="y", pct=0.999):
    y = df[ycol]
    v = y.quantile(pct)
    return df[y < v]
