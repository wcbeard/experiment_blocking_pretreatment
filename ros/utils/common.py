import logging
from functools import partial
from typing import List, Union

import pandas as pd  # type: ignore

vstack = partial(pd.concat, ignore_index=True)
hstack = partial(pd.concat, ignore_index=False, axis=1)
plot_wrap = lambda y: pd.DataFrame(dict(y=y, g=1))


def drop_outliers(df, ycol="y", pct=0.999):
    y = df[ycol]
    v = y.quantile(pct)
    return df[y < v]


def disable_logging(lib_name: Union[str, List[str]]):
    if isinstance(lib_name, list):
        for ln in lib_name:
            disable_logging(ln)
        return
    logger = logging.getLogger(lib_name)
    logger.propagate = False
    logger.setLevel(logging.ERROR)
