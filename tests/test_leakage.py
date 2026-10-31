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


def _synth_funds(stocks, seed=0, poison_from_year=None):
    rng = np.random.default_rng(seed)
    out = {}
    for s in stocks:
        yrs = list(range(2008, 2022))
        df = pd.DataFrame({"ni": rng.normal(50, 20, len(yrs)), "rev": rng.uniform(300, 500, len(yrs)),
                           "eps": rng.normal(1, 0.3, len(yrs)), "equity": rng.uniform(200, 400, len(yrs)),
                           "assets": rng.uniform(500, 900, len(yrs)), "liab": rng.uniform(100, 300, len(yrs))}, index=yrs)
        if poison_from_year is not None:
            m = df.index >= poison_from_year
            df.loc[m, :] = np.random.default_rng(1234).normal(1e6, 1e5, (m.sum(), df.shape[1]))
        out[s] = df
    return out


def test_fundamental_agent_uses_only_published_years(small_cfg, tmp_path):
    """Statements for fiscal years not yet published at decision date d (Dec 31 Y + lag > d) must not affect the score."""
    from finverse.agents.fundamental import FundamentalRuleAgent
    A = tmp_path / "a"; make_synth(A)
    cfg = override(small_cfg, **{"data.dir": str(A)})
    ds = build_dataset(cfg)
    d = ds.rebal[(ds.rebal >= "2017-06-01")][0]
    lag = 9
    ag = FundamentalRuleAgent("unused", lag, funds=_synth_funds(ds.stocks))
    y = ag._latest_year(d)
    # poison every year after the latest published one
    ag2 = FundamentalRuleAgent("unused", lag, funds=_synth_funds(ds.stocks, poison_from_year=y + 1))
    pd.testing.assert_series_equal(ag.score(ds, d).score, ag2.score(ds, d).score)
    # and the year itself is NOT available 1 day before Dec 31 + lag
    assert pd.Timestamp(year=y, month=12, day=31) + pd.DateOffset(months=lag) <= d
    assert pd.Timestamp(year=y + 1, month=12, day=31) + pd.DateOffset(months=lag) > d


def test_news_agent_ignores_same_day_and_future_articles(small_cfg, tmp_path):
    """Only articles from EARLIER calendar days than the decision date may affect the news score."""
    from finverse.agents.news import NewsSentimentAgent
    A = tmp_path / "a"; make_synth(A)
    cfg = override(small_cfg, **{"data.dir": str(A)})
    ds = build_dataset(cfg)
    d = ds.rebal[(ds.rebal >= "2017-06-01")][0]
    rng = np.random.default_rng(0)
    rows = [{"company": s, "ts": d - pd.Timedelta(days=int(rng.integers(1, 25))), "sentiment": float(rng.uniform(-1, 1))}
            for s in ds.stocks for _ in range(3)]
    base = pd.DataFrame(rows)
    leak = pd.concat([base, pd.DataFrame([{"company": s, "ts": d + pd.Timedelta(hours=h), "sentiment": 1.0 if i % 2 else -1.0}
                                          for i, s in enumerate(ds.stocks) for h in (0, 5, 48)])], ignore_index=True)
    a0 = NewsSentimentAgent(base).score(ds, d).score
    a1 = NewsSentimentAgent(leak).score(ds, d).score
    pd.testing.assert_series_equal(a0, a1)


def test_news_agent_matches_file_style_company_names(small_cfg, tmp_path):
    """Sentiment files label companies like '01. Palm Hills'; dataset stocks look like 'Palm_Hills'."""
    from finverse.agents.news import NewsSentimentAgent
    A = tmp_path / "a"; make_synth(A)
    ds = build_dataset(override(small_cfg, **{"data.dir": str(A)}))
    d = ds.rebal[(ds.rebal >= "2017-06-01")][0]
    rng = np.random.default_rng(1)
    rows = [{"company": f"{i + 1:02d}. " + s.replace("_", " "), "ts": d - pd.Timedelta(days=3 + j),
             "sentiment": float(rng.uniform(-1, 1))} for i, s in enumerate(ds.stocks) for j in range(2)]
    out = NewsSentimentAgent(pd.DataFrame(rows)).score(ds, d)
    assert out.score.notna().sum() >= len(ds.stocks) // 2
