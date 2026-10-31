"""Dataset: eligibility, rebalance dates, features at rebalance dates, forward labels."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .loader import Panel, load_panel
from .features import compute_features, FEATURE_NAMES


@dataclass
class Dataset:
    panel: Panel
    start: pd.Timestamp
    lag: int
    rebal: pd.DatetimeIndex
    elig: pd.DataFrame          # daily eligibility (bool)
    feats: dict                 # name -> DataFrame (rebal x stocks), cross-sectionally rank-normalised, NaN where ineligible
    fwd_ret: pd.DataFrame       # rebal x stocks, return from trade date of r to trade date of next rebalance
    label: pd.DataFrame         # rebal x stocks, cross-sectional rank percentile of fwd_ret (eligible only)
    label_end: pd.Series        # rebal -> date when the label is realized
    mkt_ret: pd.Series          # daily equal-weight market return (eligible stocks)
    daily_ret: pd.DataFrame

    @property
    def stocks(self):
        return list(self.panel.close.columns)


def eligibility(panel: Panel, min_history_days, max_imputed_share_60d) -> pd.DataFrame:
    real = (panel.raw_close.notna() & ~panel.imputed).cumsum()
    imp_share = panel.imputed.astype(float).rolling(60, min_periods=1).mean()
    ok = (real >= min_history_days) & panel.close.notna() & (imp_share <= max_imputed_share_60d)
    return ok


def month_end_dates(cal: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(cal, index=cal)
    return pd.DatetimeIndex(s.groupby([cal.year, cal.month]).max().values)


def build_dataset(cfg: dict, panel: Panel | None = None) -> Dataset:
    d = cfg["data"]
    if panel is None:
        panel = load_panel(d["dir"], end=d.get("end"), calendar_min_coverage=d["calendar_min_coverage"],
                           max_ffill_days=d["max_ffill_days"])
    cal = panel.calendar
    start = pd.Timestamp(d["start"])
    lag = int(cfg["backtest"]["exec_lag_days"])
    elig = eligibility(panel, d["min_history_days"], d["max_imputed_share_60d"])
    daily_ret = panel.close.pct_change(fill_method=None)
    # rebalance dates: month ends from `start`; features use history before start (warm-up)
    me = month_end_dates(cal)
    rebal = me[me >= start]
    # last month-end may be an incomplete month; drop it if it is not the true end of the data month
    if len(rebal) and rebal[-1] == cal[-1] and cal[-1] < cal[-1] + pd.offsets.MonthEnd(0) - pd.Timedelta(days=5):
        rebal = rebal[:-1]
    feats_daily = compute_features(panel.close, panel.volume)
    el_at = elig.reindex(rebal)
    feats = {}
    for name, df in feats_daily.items():
        x = df.reindex(rebal).where(el_at)
        feats[name] = x.rank(axis=1, pct=True)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    idx = pos.reindex(rebal).values
    n = len(cal)
    close = panel.close.values
    fwd = np.full((len(rebal), close.shape[1]), np.nan)
    lend = []
    for k in range(len(rebal)):
        a = idx[k] + lag
        if k + 1 < len(rebal):
            b = idx[k + 1] + lag
        else:
            b = n  # not available
        if a < n and b < n:
            fwd[k] = close[b] / close[a] - 1.0
            lend.append(cal[b])
        else:
            lend.append(pd.NaT)
    fwd_ret = pd.DataFrame(fwd, index=rebal, columns=panel.close.columns).where(el_at)
    label = fwd_ret.rank(axis=1, pct=True)
    label_end = pd.Series(lend, index=rebal)
    mkt = daily_ret.where(elig).mean(axis=1)
    return Dataset(panel=panel, start=start, lag=lag, rebal=rebal, elig=elig, feats=feats, fwd_ret=fwd_ret,
                   label=label, label_end=label_end, mkt_ret=mkt, daily_ret=daily_ret)
