"""Annual fundamentals from the TradeXAI files. No publication dates exist in these files, so availability is
modelled with a publication lag (config, PLACEHOLDER, with a sensitivity grid)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

INCOME = ["Net Income", "Total Revenues", "Basic EPS - Continuing Operations"]
BALANCE = ["Total Equity", "Total Assets", "Total Liabilities"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("%", "").str.replace("+", "", regex=False),
                         errors="coerce")


def _load_one(path: Path, items) -> pd.DataFrame:
    x = pd.read_csv(path).set_index("Item")
    yrs = [c for c in x.columns if str(c).isdigit()]
    x = x.loc[x.index.isin(items), yrs].apply(_num)
    x = x[~x.index.duplicated(keep="first")]
    x.columns = [int(c) for c in x.columns]
    return x.T  # index year


def load_fundamentals(fdir, companies) -> dict:
    """Return {company: DataFrame(year x [ni, rev, eps, equity, assets, liabilities])}."""
    fdir = Path(fdir)
    m = json.load(open(fdir / "company_ticker_mapping.json"))
    out = {}
    for c in companies:
        t = m.get(c)
        if t is None:
            continue
        fi = fdir / "income_statements" / f"{t}_incomeStatement.csv"
        fb = fdir / "balance_sheets" / f"{t}_balanceSheet.csv"
        if not fi.exists() or not fb.exists():
            continue
        inc, bal = _load_one(fi, INCOME), _load_one(fb, BALANCE)
        df = inc.join(bal, how="outer")
        df = df.rename(columns={"Net Income": "ni", "Total Revenues": "rev", "Basic EPS - Continuing Operations": "eps",
                                "Total Equity": "equity", "Total Assets": "assets", "Total Liabilities": "liab"})
        for k in ["ni", "rev", "eps", "equity", "assets", "liab"]:
            if k not in df:
                df[k] = np.nan
        out[c] = df.sort_index()
    return out
