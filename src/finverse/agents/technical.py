"""Technical ML agent: bootstrap ensemble of gradient-boosted trees predicting next-period cross-sectional rank."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .base import Agent, AgentOutput
from ..data.features import FEATURE_NAMES


def _make_model(cfg, seed):
    return HistGradientBoostingRegressor(
        max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"], l2_regularization=1.0, random_state=seed)


def _matrix(ds, date, cols):
    X = pd.DataFrame({c: ds.feats[c].loc[date] for c in cols})
    return X


class TechnicalMLAgent(Agent):
    name = "technical_ml"

    def __init__(self, cfg: dict, seed: int = 7):
        self.cfg = cfg
        self.seed = seed
        self.cols = list(FEATURE_NAMES)
        self.models = []

    def fit(self, ds, train_dates):
        Xs, ys = [], []
        for d in train_dates:
            X = _matrix(ds, d, self.cols)
            y = ds.label.loc[d]
            m = X.notna().any(axis=1) & y.notna()
            if m.sum() < 5:
                continue
            Xs.append(X[m].fillna(0.5))
            ys.append(y[m])
        self.models = []
        if not Xs:
            return
        X = pd.concat(Xs).values
        y = pd.concat(ys).values
        rng = np.random.default_rng(self.seed)
        for i in range(self.cfg["n_models"]):
            idx = rng.integers(0, len(X), len(X))
            mdl = _make_model(self.cfg, self.seed + i)
            mdl.fit(X[idx], y[idx])
            self.models.append(mdl)

    def _predict(self, ds, date):
        X = _matrix(ds, date, self.cols)
        ok = ds.elig.loc[date] & X.notna().any(axis=1)
        preds = np.full((len(self.models), len(X)), np.nan)
        if ok.sum():
            Xv = X[ok].fillna(0.5).values
            for i, m in enumerate(self.models):
                preds[i, ok.values] = m.predict(Xv)
        return X, ok, preds

    def score(self, ds, date) -> AgentOutput:
        X, ok, preds = self._predict(ds, date)
        idx = X.index
        if not self.models:
            nan = pd.Series(np.nan, index=idx)
            return AgentOutput(self.name, date, nan, nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = pd.Series(np.nanmean(preds, axis=0), index=idx)
            std = pd.Series(np.nanstd(preds, axis=0), index=idx)
        score = mean.where(ok).rank(pct=True)
        conf = 1.0 - std.where(ok).rank(pct=True)   # relative model agreement (heuristic, not a probability)
        return AgentOutput(self.name, date, score, conf, source_ids=[f"prices@{date.date()}"])

    def drivers(self, ds, date, stock, top=3):
        """Occlusion: replace one feature by the neutral rank 0.5; report the change in the stock's mean prediction."""
        if not self.models:
            return []
        X = _matrix(ds, date, self.cols)
        if stock not in X.index or X.loc[stock].isna().all():
            return []
        x = X.loc[stock].fillna(0.5).values.reshape(1, -1)
        base = np.mean([m.predict(x)[0] for m in self.models])
        out = []
        for j, c in enumerate(self.cols):
            x2 = x.copy()
            x2[0, j] = 0.5
            alt = np.mean([m.predict(x2)[0] for m in self.models])
            out.append((c, float(base - alt), float(x[0, j])))
        out.sort(key=lambda t: -abs(t[1]))
        return [{"feature": c, "effect": e, "rank_value": v} for c, e, v in out[:top]]
