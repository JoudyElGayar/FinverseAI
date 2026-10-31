import numpy as np
import pandas as pd
from conftest import make_synth
from finverse.config import override
from finverse.data.dataset import build_dataset
from finverse.cli import build_agents
from finverse.backtest.engine import run_agents, Runner, select_test_dates


def test_costs_and_returns_accounting(small_cfg, tmp_path):
    make_synth(tmp_path / "a")
    cfg = override(small_cfg, **{"data.dir": str(tmp_path / "a")})
    ds = build_dataset(cfg)
    td = select_test_dates(ds, cfg)[:8]
    ap = run_agents(ds, cfg, build_agents(cfg), td)
    r = Runner(cfg, ds, ap)
    free = r.run("equal_weight", 0.0).returns
    paid = r.run("equal_weight", 100.0).returns
    np.testing.assert_allclose(free["gross"], paid["gross"])
    np.testing.assert_allclose(paid["cost"], paid["turnover"] * 0.01)
    # first build pays full cost on invested weight
    assert abs(paid["turnover"].iloc[0] - 1.0) < 1e-9
    # equal-weight return equals mean of eligible forward returns
    d = td[3]
    el = ds.elig.loc[d]
    exp = ds.fwd_ret.loc[d][el].mean()
    assert abs(free.loc[d, "gross"] - exp) < 1e-9
    # buy and hold: no turnover after the first build
    bh = r.run("buy_hold", 25.0).returns
    assert bh["turnover"].iloc[1:].max() < 1e-9
