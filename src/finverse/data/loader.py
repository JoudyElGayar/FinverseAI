"""Load cleaned per-company CSVs into wide daily panels."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Panel:
    close: pd.DataFrame      # forward-filled (limited) close, dates x stocks
    raw_close: pd.DataFrame  # close as observed (NaN where no row)
    volume: pd.DataFrame
    imputed: pd.DataFrame    # bool
    calendar: pd.DatetimeIndex


def load_panel(data_dir, start=None, end=None, calendar_min_coverage=0.5, max_ffill_days=5) -> Panel:
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no csv files in {data_dir}")
    close, vol, imp = {}, {}, {}
    for f in files:
        df = pd.read_csv(f, parse_dates=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
        name = f.stem
        close[name] = df["Price"]
        vol[name] = df["Volume"]
        imp[name] = df["imputed"].astype(bool) if "imputed" in df else pd.Series(False, index=df.index)
    raw = pd.DataFrame(close).sort_index()
    volume = pd.DataFrame(vol).reindex(raw.index)
    imputed = pd.DataFrame(imp).reindex(raw.index).fillna(False).astype(bool)

    # trading calendar: dates where enough of the already-listed stocks have a row
    listed = raw.notna().cummax()
    coverage = raw.notna().sum(axis=1) / listed.sum(axis=1).clip(lower=1)
    # keep warm-up history before the first rebalance date: the calendar is not trimmed at `start`
    full_cal = raw.index[coverage >= calendar_min_coverage]
    if end is not None:
        full_cal = full_cal[full_cal <= pd.Timestamp(end)]
    raw = raw.reindex(full_cal)
    volume = volume.reindex(full_cal)
    imputed = imputed.reindex(full_cal).fillna(False)
    close_ff = raw.ffill(limit=max_ffill_days)
    return Panel(close=close_ff, raw_close=raw, volume=volume, imputed=imputed, calendar=full_cal)
