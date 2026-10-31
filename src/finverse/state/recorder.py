"""Write a finished run into the state store (agent outputs, regime, coordinator decisions, portfolio)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _clean(s: pd.Series) -> dict:
    return {k: (None if pd.isna(v) else float(v)) for k, v in s.items()}


def record_run(store, run_id, ap, result):
    for a in ap.scores:
        for d in ap.dates:
            store.put(run_id, d.date(), "agent_output", a,
                      {"agent": a, "as_of": str(d.date()), "score": _clean(ap.scores[a].loc[d]),
                       "confidence": _clean(ap.confs[a].loc[d])},
                      source_ids=ap.sources.get(a, {}).get(d, []))
    for d in ap.dates:
        store.put(run_id, d.date(), "regime", "hmm", {"filtered_prob": ap.regime_prob.loc[d].to_dict(),
                                                      "hard_state": int(ap.regime_hard.loc[d])})
    for d, st in result.state.items():
        store.put(run_id, d.date(), "coordinator", "dynamic",
                  {"agent_weights": st["agent_weights"], "info": {k: v for k, v in st["info"].items()},
                   "combined": _clean(st["combined"])})
        store.put(run_id, d.date(), "portfolio", result.name, {"target_weights": _clean(result.weights.loc[d])})
    store.commit()
