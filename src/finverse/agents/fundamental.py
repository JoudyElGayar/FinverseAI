"""Fundamental rule agent: transparent equal-weight composite of cross-sectional factor ranks.
Factors and weights are fixed a priori (not fitted). Statements for fiscal year Y are treated as available only
from 31 Dec Y + publication_lag_months (PLACEHOLDER: the source files carry no announcement dates)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Agent, AgentOutput
from ..data.fundamentals import load_fundamentals

FACTORS = {"roe": +1, "roa": +1, "net_margin": +1, "ni_growth": +1, "leverage": -1, "earnings_yield": +1}


class FundamentalRuleAgent(Agent):
    name = "fundamental_rule"

    def __init__(self, fdir, lag_months=9, funds=None):
        self.fdir, self.lag = fdir, int(lag_months)
        self._funds = funds
        self._cache = {}

    def _load(self, ds):
        if self._funds is None:
            self._funds = load_fundamentals(self.fdir, ds.stocks)
        return self._funds

    def _latest_year(self, d):
        # year Y is available at d if Dec 31 Y + lag months <= d
        for y in range(d.year, d.year - 4, -1):
            if pd.Timestamp(year=y, month=12, day=31) + pd.DateOffset(months=self.lag) <= d:
                return y
        return None

    def factors(self, ds, d) -> pd.DataFrame:
        funds = self._load(ds)
        y = self._latest_year(d)
        rows = {}
        px = ds.panel.close.loc[d]
        for s in ds.stocks:
            f = funds.get(s)
            r = dict.fromkeys(FACTORS, np.nan)
            if f is not None and y is not None and y in f.index:
                cur = f.loc[y]
                prev = f.loc[y - 1] if (y - 1) in f.index else None
                if cur["equity"] and cur["equity"] > 0:
                    r["roe"] = cur["ni"] / cur["equity"]
                if cur["assets"] and cur["assets"] > 0:
                    r["roa"] = cur["ni"] / cur["assets"]
                    r["leverage"] = cur["liab"] / cur["assets"]
                if cur["rev"] and cur["rev"] > 0:
                    r["net_margin"] = cur["ni"] / cur["rev"]
                if prev is not None and pd.notna(prev["ni"]) and prev["ni"] != 0:
                    r["ni_growth"] = float(np.clip((cur["ni"] - prev["ni"]) / abs(prev["ni"]), -1, 1))
                if pd.notna(cur["eps"]) and pd.notna(px[s]) and px[s] > 0:
                    r["earnings_yield"] = cur["eps"] / px[s]
            rows[s] = r
        return pd.DataFrame(rows).T

    def score(self, ds, date) -> AgentOutput:
        F = self.factors(ds, date)
        el = ds.elig.loc[date]
        F = F.where(el, np.nan)
        R = pd.DataFrame({k: (F[k].rank(pct=True) if sgn > 0 else 1 - F[k].rank(pct=True)) for k, sgn in FACTORS.items()})
        n_avail = R.notna().sum(axis=1)
        s = R.mean(axis=1).where(n_avail >= 3).rank(pct=True)
        conf = (n_avail / len(FACTORS)).where(s.notna())
        return AgentOutput(self.name, date, s, conf, source_ids=[f"fundamentals<=FY{self._latest_year(date)}"])

    def drivers(self, ds, date, stock, top=3):
        F = self.factors(ds, date)
        r = F.loc[stock]
        out = []
        for k, sgn in FACTORS.items():
            if pd.notna(r[k]):
                col = F[k].where(ds.elig.loc[date])
                rk = float(col.rank(pct=True)[stock]) if pd.notna(col[stock]) else np.nan
                out.append({"feature": k, "effect": None, "rank_value": rk if sgn > 0 else 1 - rk})
        out.sort(key=lambda t: -abs((t["rank_value"] if t["rank_value"] == t["rank_value"] else 0.5) - 0.5))
        return out[:top]
