"""Leakage tests. The future-poisoning test is the most important test in the repo."""
import numpy as np
import pandas as pd
from conftest import make_synth
from finverse.config import override
from finverse.data.dataset import build_dataset
from finverse.cli import build_agents
from finverse.backtest.engine import run_agents, Runner, select_test_dates


def _pipeline(cfg, d, dates):
    cfg = override(cfg, **{"data.dir": str(d)})
    ds = build_dataset(cfg)
    ap = run_agents(ds, cfg, build_agents(cfg), dates)
    r = Runner(cfg, ds, ap)
    res = r.run("dynamic", 25.0, keep_state=True)
    return ds, ap, res


def test_future_poisoning(small_cfg, tmp_path):
    """Replacing every price after day T must not change any score, agent weight or target weight up to T."""
    A, B = tmp_path / "a", tmp_path / "b"
    make_synth(A); T = "2018-09-30"
    make_synth(B, poison_after=T)
    cfg = override(small_cfg, **{"data.dir": str(A)})
    ds0 = build_dataset(cfg)
    dates = ds0.rebal[(ds0.rebal >= "2016-01-01") & (ds0.rebal <= pd.Timestamp(T))]
    dates = dates[dates.year >= 2016]
    dsA, apA, resA = _pipeline(small_cfg, A, dates)
    dsB, apB, resB = _pipeline(small_cfg, B, dates)
    for a in apA.scores:
        pd.testing.assert_frame_equal(apA.scores[a], apB.scores[a], check_exact=False, atol=1e-9)
    pd.testing.assert_frame_equal(apA.regime_prob, apB.regime_prob, check_exact=False, atol=1e-9)
    pd.testing.assert_frame_equal(resA.weights, resB.weights, check_exact=False, atol=1e-6)
    for d in dates:
        wa, wb = resA.state[d]["agent_weights"], resB.state[d]["agent_weights"]
        for k in wa:
            assert abs(wa[k] - wb[k]) < 1e-9, (d, k)


def test_purge_training_labels_realized(small_cfg, tmp_path):
    A = tmp_path / "a"; make_synth(A)
    cfg = override(small_cfg, **{"data.dir": str(A)})
    ds = build_dataset(cfg)
    td = select_test_dates(ds, cfg)
    for y in sorted(set(td.year)):
        first = td[td.year == y][0]
        train = [d for d in ds.rebal if d >= ds.start and pd.notna(ds.label_end[d]) and ds.label_end[d] <= first]
        assert train, y
        assert max(ds.label_end[d] for d in train) <= first


def test_dev_labels_do_not_reach_holdout(small_cfg, tmp_path):
    A = tmp_path / "a"; make_synth(A)
    cfg = override(small_cfg, **{"data.dir": str(A)})
    ds = build_dataset(cfg)
    td = select_test_dates(ds, cfg)
    hold = pd.Timestamp(cfg["backtest"]["final_holdout_start"])
    assert (ds.label_end.reindex(td) < hold).all()
    assert td.max() < hold


def test_features_are_causal(small_cfg, tmp_path):
    """Features at date t computed on data truncated at t equal those computed on the full data."""
    from finverse.data.features import compute_features
    A = tmp_path / "a"; make_synth(A, n=4)
    from finverse.data.loader import load_panel
    p = load_panel(A)
    full = compute_features(p.close, p.volume)
    t = p.calendar[2500]
    cut = compute_features(p.close.loc[:t], p.volume.loc[:t])
    for k in full:
        pd.testing.assert_series_equal(full[k].loc[t], cut[k].loc[t], check_names=False, atol=1e-9)
