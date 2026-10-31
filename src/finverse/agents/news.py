"""News agent: decayed average of per-article sentiment over the prior window, ranked across stocks.

Causality rule: an article counts for decision date d only if it was published on an EARLIER calendar day than d.
(The source files do not document a timezone, so same-day articles are excluded to be safe.)
Stocks with no article in the window are not scored (NaN); the coordinator ignores missing scores.
Sentiment per article comes from a precomputed file (scripts/score_news_finbert.py) so the agent itself is repeatable."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Agent, AgentOutput


class NewsSentimentAgent(Agent):
    name = "news_sentiment"

    def __init__(self, sentiment: pd.DataFrame, window_days=30, halflife_days=14, full_conf_articles=3):
        """sentiment: DataFrame[company, ts, sentiment] with sentiment in [-1, 1] (positive minus negative probability)."""
        s = sentiment.copy()
        s["ts"] = pd.to_datetime(s["ts"])
        self.s = s.sort_values("ts").reset_index(drop=True)
        self.window, self.half, self.n_full = window_days, halflife_days, full_conf_articles

    def _stats(self, date):
        d0 = pd.Timestamp(date).normalize()
        m = (self.s["ts"] < d0) & (self.s["ts"] >= d0 - pd.Timedelta(days=self.window))
        w = self.s.loc[m].copy()
        if w.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        age = (d0 - w["ts"]).dt.total_seconds() / 86400.0
        w["w"] = 0.5 ** (age / self.half)
        g = w.groupby("company")
        mean = (w["w"] * w["sentiment"]).groupby(w["company"]).sum() / g["w"].sum()
        return mean, g.size()

    def score(self, ds, date) -> AgentOutput:
        mean, cnt = self._stats(date)
        idx = pd.Index(ds.stocks)
        s = mean.reindex(idx).where(ds.elig.loc[date])
        score = s.rank(pct=True)
        conf = (cnt.reindex(idx) / self.n_full).clip(upper=1.0).where(score.notna())
        return AgentOutput(self.name, date, score, conf, source_ids=[f"news<{pd.Timestamp(date).date()}"])

    def drivers(self, ds, date, stock, top=3):
        d0 = pd.Timestamp(date).normalize()
        m = (self.s["company"] == stock) & (self.s["ts"] < d0) & (self.s["ts"] >= d0 - pd.Timedelta(days=self.window))
        w = self.s.loc[m].copy()
        if w.empty:
            return []
        w["a"] = w["sentiment"].abs()
        w = w.sort_values("a", ascending=False).head(top)
        return [{"feature": f"article {r.ts.date()}", "effect": float(r.sentiment), "rank_value": float("nan")} for r in w.itertuples()]
