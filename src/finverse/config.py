from __future__ import annotations
from pathlib import Path
import copy
import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override(cfg: dict, **kv) -> dict:
    """Return a deep copy with dotted-key overrides, e.g. override(cfg, **{'backtest.cost_bps_one_way': 50})."""
    out = copy.deepcopy(cfg)
    for k, v in kv.items():
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = v
    return out
