"""Reliability- and regime-aware agent weighting (uses only outcomes realized by the decision date)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def ew_mean(vals: np.ndarray, ages: np.ndarray, halflife: float) -> float:
    w = 0.5 ** (ages / halflife)
    return float((w * vals).sum() / w.sum())


def softmax(x, k):
    z = k * (np.asarray(x) - np.max(x))
    e = np.exp(z)
    return e / e.sum()


def agent_weights(agents, ic: pd.DataFrame, regime_hard: pd.Series, label_end: pd.Series, k: int, ccfg: dict,
                  scheme: str = "dynamic"):
    """Weights for test date index k.

    ic: DataFrame (test dates x agents) of rank-IC between the score at date j and its realized label.
    Only rows j<k with label_end[j] <= date k are used (realized outcomes).
    Returns (weights dict, info dict).
    """
    names = list(agents)
    now = ic.index[k]
    eq = {a: 1.0 / len(names) for a in names}
    if scheme == "equal":
        return eq, {"reason": "equal"}
    if scheme == "fixed":
        fw = ccfg.get("fixed_weights") or {}
        tot = sum(fw.get(a, 0.0) for a in names)
        return ({a: fw.get(a, 0.0) / tot for a in names} if tot > 0 else eq), {"reason": "fixed"}
    if scheme.startswith("oracle"):   # analysis only: uses the realized IC of the *current* period (future info)
        cur = ic.iloc[k].fillna(0.0).values
        return dict(zip(names, softmax(cur, ccfg["softmax_k"]))), {"reason": "oracle_ANALYSIS_ONLY"}
    usable = [j for j in range(k) if pd.notna(label_end.iloc[j]) and label_end.iloc[j] <= now]
    if len(usable) < ccfg["min_obs"]:
        return eq, {"reason": "burn_in", "n_obs": len(usable)}
    ages = np.array([k - j for j in usable], float)
    rel = {}
    cur_reg = regime_hard.iloc[k]
    info = {"n_obs": len(usable), "regime": int(cur_reg)}
    reg_idx = [j for j in usable if regime_hard.iloc[j] == cur_reg]
    for a in names:
        v = ic[a].iloc[usable].values
        m = np.isfinite(v)
        overall = ew_mean(v[m], ages[m], ccfg["halflife_months"]) if m.any() else 0.0
        if len(reg_idx) >= ccfg["min_obs_regime"]:
            vr = ic[a].iloc[reg_idx].values
            ar = np.array([k - j for j in reg_idx], float)
            mr = np.isfinite(vr)
            reg = ew_mean(vr[mr], ar[mr], ccfg["halflife_months"]) if mr.any() else overall
            rel[a] = (1 - ccfg["regime_blend"]) * overall + ccfg["regime_blend"] * reg
        else:
            rel[a] = overall
    w = softmax([rel[a] for a in names], ccfg["softmax_k"])
    info["reliability"] = rel
    info["n_obs_regime"] = len(reg_idx)
    return dict(zip(names, w)), info


def combine(scores: dict, confs: dict, weights: dict) -> pd.Series:
    num = None; den = None
    for a, w in weights.items():
        s = scores[a]; c = 0.25 + 0.75 * confs[a].fillna(0.0)
        m = s.notna()
        wc = (w * c).where(m, 0.0)
        term = (wc * s.fillna(0.0))
        num = term if num is None else num + term
        den = wc if den is None else den + wc
    out = num / den.where(den > 0)
    return out
