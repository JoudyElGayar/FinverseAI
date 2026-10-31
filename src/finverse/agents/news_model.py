"""News model agent: a small ridge regression on article-sentiment features, trained walk-forward.

Training uses only decision dates whose labels are realized before the fold starts (the engine passes those dates),
and features use only articles from earlier calendar days than each decision date. Stocks with no article in the
prior 30 days are not scored. Sentiment per article comes from the pre-trained FinBERT file (no training on our labels)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .base import Agent, AgentOutput
from ..data.news import _norm

FEATURES = ["mean_30d", "mean_7d", "log_count_30d", "change_vs_90d"]


class NewsModelAgent(Agent):
    name = "news_model"

    def __init__(self, sentiment: pd.DataFrame, alpha=10.0, min_train_rows=100, full_conf_articles=3):
        s = sentiment.copy()
        s["ts"] = pd.to_datetime(s["ts"]); s["company"] = s["company"].map(_norm)
        self.s = s.sort_values("ts").reset_index(drop=True)
        self.alpha, self.min_rows, self.n_full = alpha, min_train_rows, full_conf_articles
        self.model = None
        self._cache = {}

    def _feats(self, ds, date) -> pd.DataFrame:
        key = pd.Timestamp(date)
        if key in self._cache:
            return self._cache[key]
        d0 = key.normalize()
        w = self.s[(self.s["ts"] < d0) & (self.s["ts"] >= d0 - pd.Timedelta(days=90))]
        keys = pd.Index([_norm(x) for x in ds.stocks])
        def grp(sub):
            g = sub.groupby("company")["sentiment"]
            return g.mean(), g.size()
        m90, _ = grp(w)
        m30, c30 = grp(w[w["ts"] >= d0 - pd.Timedelta(days=30)])
        m7, _ = grp(w[w["ts"] >= d0 - pd.Timedelta(days=7)])
        F = pd.DataFrame(index=pd.Index(ds.stocks))
        F["mean_30d"] = m30.reindex(keys).values
        F["count_30d"] = c30.reindex(keys).values
        F["mean_7d"] = m7.reindex(keys).values
        F["mean_90d"] = m90.reindex(keys).values
        F["mean_7d"] = F["mean_7d"].fillna(F["mean_30d"])
        F["log_count_30d"] = np.log1p(F["count_30d"])
        F["change_vs_90d"] = F["mean_30d"] - F["mean_90d"]
        self._cache[key] = F
        return F

    def fit(self, ds, train_dates):
        Xs, ys = [], []
        for d in train_dates:
            F = self._feats(ds, d)
            y = ds.label.loc[d]
            m = F["mean_30d"].notna() & y.notna() & ds.elig.loc[d]
            if m.sum() >= 3:
                Xs.append(F.loc[m, FEATURES]); ys.append(y[m])
        self.model = None
        if not Xs:
            return
        X = pd.concat(Xs); y = pd.concat(ys)
        if len(X) < self.min_rows:
            return
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X.values, y.values)

    def score(self, ds, date) -> AgentOutput:
        idx = pd.Index(ds.stocks)
        nan = pd.Series(np.nan, index=idx)
        if self.model is None:
            return AgentOutput(self.name, date, nan, nan)
        F = self._feats(ds, date)
        ok = F["mean_30d"].notna() & ds.elig.loc[date]
        pred = pd.Series(np.nan, index=idx)
        if ok.any():
            pred[ok] = self.model.predict(F.loc[ok, FEATURES].values)
        conf = (F["count_30d"] / self.n_full).clip(upper=1.0).where(pred.notna())
        return AgentOutput(self.name, date, pred.rank(pct=True), conf, source_ids=[f"news<{pd.Timestamp(date).date()}"])

    def drivers(self, ds, date, stock, top=3):
        if self.model is None:
            return []
        F = self._feats(ds, date)
        if stock not in F.index or pd.isna(F.loc[stock, "mean_30d"]):
            return []
        coef = self.model[-1].coef_; sc = self.model[0]
        z = sc.transform(F.loc[[stock], FEATURES].values)[0]
        out = sorted(zip(FEATURES, coef * z, F.loc[stock, FEATURES].values), key=lambda t: -abs(t[1]))[:top]
        return [{"feature": f, "effect": float(e), "rank_value": float("nan")} for f, e, v in out]
