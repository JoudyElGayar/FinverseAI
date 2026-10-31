"""Causal daily features (every value at date t uses data up to and including t)."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_NAMES = ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_1", "vol_1m", "vol_3m", "dd_12m",
                 "ma200_dist", "rsi14", "macd_hist", "dvol_ratio", "log_dvol"]


def compute_features(close: pd.DataFrame, volume: pd.DataFrame) -> dict:
    c = close
    r = c.pct_change(fill_method=None)
    f = {}
    f["mom_1m"] = c / c.shift(21) - 1
    f["mom_3m"] = c / c.shift(63) - 1
    f["mom_6m"] = c / c.shift(126) - 1
    f["mom_12m"] = c / c.shift(252) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["vol_1m"] = r.rolling(21, min_periods=15).std()
    f["vol_3m"] = r.rolling(63, min_periods=40).std()
    f["dd_12m"] = c / c.rolling(252, min_periods=150).max() - 1
    f["ma200_dist"] = c / c.rolling(200, min_periods=150).mean() - 1
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    dn = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = up / dn.where(dn > 0)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((dn == 0) & (up > 0)), 100.0)
    f["rsi14"] = rsi
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (macd - sig) / c
    dv = c * volume
    f["dvol_ratio"] = dv.rolling(21, min_periods=15).mean() / dv.rolling(126, min_periods=80).mean()
    f["log_dvol"] = np.log1p(dv.rolling(63, min_periods=40).mean())
    for k in f:
        f[k] = f[k].replace([np.inf, -np.inf], np.nan)
    return f
