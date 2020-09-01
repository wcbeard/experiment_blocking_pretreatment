import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def mk_bootstrap_df(bs_dct, n_reps=1_000):
    dfs = [
        pd.DataFrame({"y": bs.draw_replicates(n_reps)}).assign(g=k)
        for k, bs in bs_dct.items()
    ]
    df = pd.concat(dfs)
    return df


def mk_bootstrap_gb(df, gb="os", stat_fn=np.mean, n_reps=10_000):
    means_ = {
        k: BootstrapStat(gdf.pre, stat=stat_fn) for k, gdf in df.groupby(gb)
    }
    mean_draws = mk_bootstrap_df(means_, n_reps=n_reps)
    return mean_draws


class BootstrapStat:
    def __init__(self, a, stat=np.median, replicate_size=None):
        self.a = np.array(a)
        self.stat = stat
        self.replicate_size = replicate_size or len(a)

    def rvs(self, n):
        return np.random.choice(self.a, size=n, replace=True)

    def rvs2d(self, n, m):
        return np.random.choice(self.a, size=(n, m), replace=True)

    def draw_replicate(self):
        samps = self.rvs(n=self.replicate_size)
        return self.stat(samps)

    def _draw_replicates_slow(self, n_reps):
        res = [self.draw_replicate() for _ in range(n_reps)]
        return np.array(res)

    def draw_replicates(self, n_reps):
        rvs = self.rvs2d(self.replicate_size, n_reps)
        res = self.stat(rvs, axis=0)
        assert len(res) == n_reps
        return res
