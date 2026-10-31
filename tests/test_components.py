import numpy as np
import pandas as pd
import pytest
from finverse.regime.hmm import GaussianHMM
from finverse.optimizer.portfolio import optimize, HAVE_CVXPY
from finverse.backtest.metrics import perf, block_bootstrap_diff


def test_hmm_recovers_regimes_and_filter_is_causal():
    rng = np.random.default_rng(0)
    s = np.r_[np.zeros(600), np.ones(300), np.zeros(600)].astype(int)
    x = np.where(s == 0, rng.normal(0.001, 0.005, len(s)), rng.normal(-0.002, 0.02, len(s)))
    h = GaussianHMM(2, 3, seed=1).fit(x[:900])
    assert h.var[1] > 4 * h.var[0]
    p = h.filter(x)
    assert p[650:850, 1].mean() > 0.8 and p[:500, 1].mean() < 0.2
    x2 = x.copy(); x2[1000:] = rng.normal(0, 0.1, len(x) - 1000)
    np.testing.assert_allclose(h.filter(x2)[:1000], p[:1000])


def _problem(n=8, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)); cov = A @ A.T / n * 0.01 + 0.001 * np.eye(n)
    mu = rng.normal(0, 0.02, n)
    return mu, cov


PC = {"risk_aversion": 5.0, "max_position": 0.25, "min_cash": 0.0, "turnover_cap": 0.4, "backend": "scipy"}


def test_optimizer_constraints():
    mu, cov = _problem()
    w0 = np.full(8, 0.05)
    w, info = optimize(mu, cov, w0, PC, cost_bps=25)
    assert info["status"] == "ok"
    assert (w >= -1e-9).all() and (w <= PC["max_position"] + 1e-6).all()
    assert w.sum() <= 1 + 1e-6
    assert np.abs(w - w0).sum() <= PC["turnover_cap"] * info["relax"] + 1e-5


def test_optimizer_full_investment():
    mu, cov = _problem()
    w, info = optimize(np.zeros(8), cov, np.zeros(8), PC, first_build=True, full=True)
    assert abs(w.sum() - 1) < 1e-5


@pytest.mark.skipif(not HAVE_CVXPY, reason="cvxpy not installed")
def test_cvxpy_matches_scipy():
    mu, cov = _problem()
    w0 = np.full(8, 0.05)
    a, _ = optimize(mu, cov, w0, PC, 25, backend="scipy")
    b, _ = optimize(mu, cov, w0, PC, 25, backend="cvxpy")
    assert np.abs(a - b).max() < 5e-3


def test_perf_and_bootstrap():
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.04, 60))
    m = perf(r)
    assert m["months"] == 60 and -1 < m["max_drawdown"] <= 0
    d = block_bootstrap_diff(r, r, reps=50)
    assert abs(d["sharpe_diff"]) < 1e-12 and d["ci_low"] <= 0 <= d["ci_high"]


def test_writer_numbers_guard():
    from finverse.explain.writer import TemplateWriter, LLMWriter, check_numbers
    e = {"date": "2020-01-31", "regime_prob": {"0": 0.7, "1": 0.3}, "agent_weights": {"a": 0.6, "b": 0.4},
         "holdings": [{"stock": "X", "weight": 0.1, "combined_score": 0.8, "agent_scores": {"a": 0.9, "b": 0.6},
                       "disagreement": 0.3, "counterfactual_weight_change_if_agent_removed": {"a": -0.02}}]}
    t = TemplateWriter().write(e)
    assert "10.0%" in t and "0.80" in t
    assert not check_numbers(t + " 99", t)
    assert LLMWriter(lambda p: "Invented 77 percent").write(e) == t
    assert LLMWriter(lambda p: p.split("\n\n", 1)[1]).write(e) == t
