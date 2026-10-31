from __future__ import annotations
import numpy as np
import pandas as pd


def perf(ret: pd.Series, rf_annual: float = 0.0, periods=12) -> dict:
    r = ret.dropna()
    if len(r) < 2:
        return {}
    rf_m = (1 + rf_annual) ** (1 / periods) - 1
    ex = r - rf_m
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    ann_ret = wealth.iloc[-1] ** (periods / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(periods)
    sharpe = ex.mean() / r.std(ddof=1) * np.sqrt(periods) if r.std(ddof=1) > 0 else np.nan
    return {"months": len(r), "ann_return": ann_ret, "ann_vol": vol, "sharpe": sharpe,
            "max_drawdown": dd.min(), "hit_rate": (r > 0).mean(), "total_return": wealth.iloc[-1] - 1}


def block_bootstrap_diff(a: pd.Series, b: pd.Series, block=3, reps=2000, seed=7, rf_annual=0.0):
    """CI for the Sharpe difference (a - b) using a circular moving-block bootstrap on paired monthly returns."""
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 2 * block:
        return {}
    x = df.values
    n = len(x)
    rng = np.random.default_rng(seed)
    rf_m = (1 + rf_annual) ** (1 / 12) - 1
    nb = int(np.ceil(n / block))

    def sh(v):
        s = v.std(ddof=1)
        return (v.mean() - rf_m) / s * np.sqrt(12) if s > 0 else np.nan

    diffs = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        smp = x[idx]
        diffs[i] = sh(smp[:, 0]) - sh(smp[:, 1])
    point = sh(x[:, 0]) - sh(x[:, 1])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"sharpe_diff": point, "ci_low": lo, "ci_high": hi, "n": n}
