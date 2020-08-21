import datetime as dt
import itertools as it
import operator as op
import os
import re
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache, partial, reduce, wraps
from glob import glob
from importlib import reload
from itertools import count
from operator import attrgetter as prop
from operator import itemgetter as sel
from os import path
from os.path import *
from pathlib import Path

import altair as A

# import dask
# import dask.dataframe as dd
import fastparquet
import matplotlib.pyplot as plt
import mystan as ms
import numpy as np
import numpy.random as nr
import pandas as pd
import pandas_utils as pu
import pandas_utils3 as p3
import plotnine as p9
import scipy as sp
import scipy.stats as sts
import seaborn as sns

# !pip install simplejson
import simplejson
import toolz.curried as z
from altair import Chart
from altair import datum as D
from altair import expr as E
from faster_pandas import MyPandas as fp
from fastparquet import ParquetFile
from joblib import Memory
from numba import njit
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal
from plotnine import aes, ggplot, ggtitle, qplot, theme, xlab, ylab
from pyarrow.parquet import ParquetFile as Paf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

ss = lambda x: StandardScaler().fit_transform(x.values[:, None]).ravel()

sns.set_palette("colorblind")
mem = Memory(location="cache", verbose=0)

# Myutils
import myutils as mu
from big_query import bq_read


reload(mu)

ap = mu.dmap


vc = z.compose(Series, Counter)


def mk_geom(p9, pref="geom_"):
    geoms = [c for c in dir(p9) if c.startswith(pref)]
    geom = lambda: None
    geom.__dict__.update(
        {name[len(pref) :]: getattr(p9, name) for name in geoms}
    )
    return geom


geom = mk_geom(p9, pref="geom_")
facet = mk_geom(p9, pref="facet_")


def run_magics():
    args = [
        "matplotlib inline",
        "autocall 1",
        "load_ext autoreload",
        "autoreload 2",
    ]
    for arg in args:
        get_ipython().magic(arg)


DataFrame.sel = lambda df, f: df[[c for c in df if f(c)]]
Path.g = lambda self, *x: list(self.glob(*x))

pd.options.display.width = 220
pd.options.display.min_rows = 40
pd.options.display.max_columns = 30

A.data_transformers.enable("json", prefix="altair/altair-data")

lrange = z.compose(list, range)
lmap = z.compose(list, map)
lfilter = z.compose(list, filter)


def read(fn, mode="r"):
    with open(fn, mode) as fp:
        txt = fp.read()
    return txt


@mem.cache
def bq_read_cache(*a, **k):
    return bq_read(*a, **k)


oss = lambda: None
oss.__dict__.update(dict(m="Mac OS X", w="Windows NT", l="Linux"))


def counts2_8020(cts, thresh=.9):
    """
    MOVE TO MYUTILS AT SOME POINT
    Find out n, where the top n elems account for `thresh`% of
    the counts.
    """
    cts = cts.sort_values(ascending=False)
    cs = cts.cumsum().pipe(lambda x: x / x.max())
    loc = cs.searchsorted(thresh)
    pct = loc / len(cs)
    ct_val = cts.iloc[loc]
    print(
        "Largest {:.1%} elems account for {:.1%} of counts (val == {})".format(
            pct, thresh, ct_val
        )
    )
    return dict(cdf_x=thresh, cdf_y=pct, val=ct_val, nth=loc)


mu.set_import_name(mu)