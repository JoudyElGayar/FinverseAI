"""Macro series with release-date handling.

File format (long CSV): series, period_end, release_date, value, source.
period_end is the period the number describes; release_date is when it became public. Only release_date matters for
the backtest: a value is usable at decision date d only if release_date < d (earlier calendar day).
If a series is later revised, add a new row with the same period_end and the revision's release_date; the as-of view
then uses the latest revision released before d. Rows with no release_date are rejected (we do not guess)."""
from __future__ import annotations
import pandas as pd

REQUIRED = ["series", "period_end", "release_date", "value"]


def load_macro(path) -> pd.DataFrame:
    d = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise ValueError(f"macro file missing columns: {missing}")
    d["period_end"] = pd.to_datetime(d["period_end"], errors="coerce")
    d["release_date"] = pd.to_datetime(d["release_date"], errors="coerce")
    bad = d[d["period_end"].isna() | d["release_date"].isna()]
    if len(bad):
        raise ValueError(f"{len(bad)} macro rows have no valid period_end/release_date; refusing to guess release dates")
    if (d["release_date"] < d["period_end"]).any():
        raise ValueError("some rows have release_date before period_end")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.dropna(subset=["value"]).sort_values(["series", "period_end", "release_date"]).reset_index(drop=True)


def macro_asof(macro: pd.DataFrame, date) -> pd.DataFrame:
    """Latest known value of each series at decision date `date`, using only releases from earlier calendar days.
    Returns DataFrame[series, period_end, release_date, value, prev_value, change]."""
    d0 = pd.Timestamp(date).normalize()
    k = macro[macro["release_date"] < d0]
    if k.empty:
        return pd.DataFrame(columns=["series", "period_end", "release_date", "value", "prev_value", "change"])
    last_rev = k.sort_values("release_date").groupby(["series", "period_end"], as_index=False).last()
    last_rev = last_rev.sort_values(["series", "period_end"])
    out = []
    for s, g in last_rev.groupby("series"):
        cur = g.iloc[-1]; prev = g.iloc[-2]["value"] if len(g) > 1 else float("nan")
        out.append({"series": s, "period_end": cur["period_end"], "release_date": cur["release_date"],
                    "value": cur["value"], "prev_value": prev, "change": cur["value"] - prev})
    return pd.DataFrame(out)
