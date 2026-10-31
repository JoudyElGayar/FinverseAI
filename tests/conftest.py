import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finverse.config import load_config, override  # noqa: E402


def make_synth(dirpath: Path, n=12, start="2008-01-01", end="2021-12-31", seed=0, poison_after=None, poison_seed=99):
    """Synthetic EGX-like price files (random walks, weekdays Sun-Thu style calendar approximated by business days)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0004, 0.01, len(dates))
    for i in range(n):
        beta = 0.6 + 0.8 * rng.random()
        eps = rng.normal(0, 0.012, len(dates))
        r = beta * mkt + eps
        if poison_after is not None:
            prng = np.random.default_rng(poison_seed + i)
            mask = dates > pd.Timestamp(poison_after)
            r = r.copy()
            r[mask] = prng.normal(0.001, 0.03, mask.sum())
        price = 10 * np.exp(np.cumsum(r))
        vol = rng.integers(10_000, 100_000, len(dates)).astype(float)
        df = pd.DataFrame({"Date": dates, "Price": price, "Open": price, "High": price, "Low": price,
                           "Volume": vol, "imputed": False})
        df.to_csv(dirpath / f"S{i:02d}.csv", index=False)


@pytest.fixture(scope="session")
def small_cfg():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    cfg = override(cfg, **{
        "data.start": "2012-01-01", "data.min_history_days": 260,
        "backtest.first_test_year": 2016, "backtest.final_holdout_start": "2020-01-01",
        "agents.technical_ml.n_models": 2, "agents.technical_ml.max_iter": 25,
        "regime.n_restarts": 2, "coordinator.min_obs": 6, "coordinator.min_obs_regime": 3,
        "backtest.bootstrap_reps": 100, "portfolio.backend": "scipy", "agents.fundamental.enabled": False})
    return cfg


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    make_synth(d)
    return d
