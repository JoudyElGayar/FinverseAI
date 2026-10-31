"""Explanation layer: numeric Bull/Bear, disagreement and counterfactuals computed from logged scores.
An LLM (later) may only rephrase these numbers; it never produces them."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..coordinator.weights import combine
from ..optimizer.portfolio import optimize, effective_mu


def _weights_from(comb, names, mu_scale, cov, w_prev, pcfg, cost_bps, first):
    mu = mu_scale * (comb[names].values - 0.5)
    w, _ = optimize(effective_mu(mu, cov, pcfg), cov, w_prev, pcfg, cost_bps, first, full=bool(pcfg.get("fully_invested", False)))
    return pd.Series(w, index=names)


def explain_date(runner, result, date, agents, top_n=5, cost_bps=25.0):
    ds, ap, cfg = runner.ds, runner.ap, runner.cfg
    st = result.state[date]
    wts = st["agent_weights"]
    comb = st["combined"]
    target = result.weights.loc[date]
    held = target[target > 0.005].sort_values(ascending=False).head(top_n)
    names = list(st["mu"].index)
    w_prev = st["w_prev"][names].values
    cov = runner.cov(date, names)
    pcfg = cfg["portfolio"]
    first = date == ap.dates[0]
    base = pd.Series(target[names].values, index=names)
    ag = {a.name: a for a in agents}
    out = {"date": str(date.date()), "regime": st["regime"], "regime_prob": st["regime_prob"],
           "agent_weights": wts, "agent_weight_info": {k: v for k, v in st["info"].items() if k != "reliability"},
           "reliability": st["info"].get("reliability"), "holdings": []}
    cfs = {}
    for a in wts:
        rest = {k: v for k, v in wts.items() if k != a}
        if not rest:
            continue
        z = sum(rest.values()); rest = {k: v / z for k, v in rest.items()}
        c2 = combine({k: ap.scores[k].loc[date] for k in rest}, {k: ap.confs[k].loc[date] for k in rest}, rest)
        cfs[a] = _weights_from(c2, names, pcfg["alpha_scale"], cov, w_prev, pcfg, cost_bps, first)
    for s, w in held.items():
        sc = {a: float(ap.scores[a].loc[date, s]) for a in wts}
        cf = {a: float(ap.confs[a].loc[date, s]) for a in wts}
        wv = np.array([wts[a] for a in wts]); sv = np.array([sc[a] for a in wts])
        mean = float((wv * sv).sum() / wv.sum())
        disagreement = float(np.sqrt((wv * (sv - mean) ** 2).sum() / wv.sum()))
        bull = [{"agent": a, "score": sc[a], "weight": wts[a]} for a in wts if sc[a] > 0.5]
        bear = [{"agent": a, "score": sc[a], "weight": wts[a]} for a in wts if sc[a] <= 0.5]
        drivers = {a: ag[a].drivers(ds, date, s) for a in wts if a in ag}
        counter = {a: float(cfs[a][s] - base[s]) for a in cfs if s in cfs[a].index}
        out["holdings"].append({"stock": s, "weight": float(w), "combined_score": float(comb[s]),
                                "agent_scores": sc, "agent_confidence": cf, "disagreement": disagreement,
                                "bull": bull, "bear": bear, "drivers": drivers,
                                "counterfactual_weight_change_if_agent_removed": counter})
    return out


def render_text(e: dict) -> str:
    lines = [f"Decision date {e['date']}. Regime state {e['regime']} "
             f"(probabilities {', '.join(f'{k}={v:.2f}' for k, v in e['regime_prob'].items())})."]
    lines.append("Agent weights: " + ", ".join(f"{a} {w:.2f}" for a, w in e["agent_weights"].items()) + ".")
    for h in e["holdings"]:
        lines.append(f"\n{h['stock']}: weight {h['weight']:.1%}, combined score {h['combined_score']:.2f}, "
                     f"agent disagreement {h['disagreement']:.2f}.")
        lines.append("  Bull: " + (", ".join(f"{b['agent']} ({b['score']:.2f})" for b in h["bull"]) or "none"))
        lines.append("  Bear: " + (", ".join(f"{b['agent']} ({b['score']:.2f})" for b in h["bear"]) or "none"))
        for a, dr in h["drivers"].items():
            if dr:
                lines.append(f"  {a} drivers: " + ", ".join(
                    f"{d['feature']} (rank {d['rank_value']:.2f}" + (f", effect {d['effect']:+.3f})" if d['effect'] is not None else ")")
                    for d in dr))
        cf = h["counterfactual_weight_change_if_agent_removed"]
        lines.append("  If removed -> weight change: " + ", ".join(f"{a} {v:+.1%}" for a, v in cf.items()))
    return "\n".join(lines)
