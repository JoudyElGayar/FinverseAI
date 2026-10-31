from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd

from .config import load_config
from .data.dataset import build_dataset
from .agents.technical import TechnicalMLAgent
from .agents.standins import REGISTRY
from .backtest.engine import select_test_dates, run_agents, Runner
from .backtest.metrics import perf, block_bootstrap_diff


def build_agents(cfg):
    ags = [TechnicalMLAgent(cfg["agents"]["technical_ml"], seed=cfg["backtest"]["seed"])]
    ags += [REGISTRY[n]() for n in cfg["agents"]["standins"]]
    fc = cfg["agents"].get("fundamental", {})
    if fc.get("enabled"):
        from .agents.fundamental import FundamentalRuleAgent
        ags.append(FundamentalRuleAgent(fc["dir"], fc["publication_lag_months"]))
    nc = cfg["agents"].get("news", {})
    if nc.get("enabled"):
        from pathlib import Path
        import pandas as pd
        from .agents.news import NewsSentimentAgent
        sf = Path(nc["sentiment_file"])
        if sf.exists():
            ags.append(NewsSentimentAgent(pd.read_csv(sf), nc["window_days"], nc["halflife_days"], nc["full_conf_articles"]))
        else:
            print(f"news agent enabled but {sf} not found: run scripts/score_news_finbert.py first (skipping News agent)")
    return ags


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def run(cfg, final=False, force=False, outdir="results", schemes=None, quiet=False):
    lock = Path(outdir) / "FINAL_LOCK"
    if final and lock.exists() and not force:
        raise SystemExit("Final holdout already run (results/FINAL_LOCK exists). Refusing: it is touched once.")
    ds = build_dataset(cfg)
    td = select_test_dates(ds, cfg, final=final)
    agents = build_agents(cfg)
    t0 = time.time()
    ap = run_agents(ds, cfg, agents, td)
    if not quiet:
        print(f"agents done in {time.time()-t0:.0f}s, {len(td)} test dates")
    runner = Runner(cfg, ds, ap)
    schemes = schemes or ["dynamic", "equal", "fixed_skip", "equal_weight", "buy_hold", "min_variance"] + \
        [f"only:{a.name}" for a in agents]
    schemes = [s for s in schemes if s != "fixed_skip"]
    bps = cfg["backtest"]["cost_bps_one_way"]
    results = {}
    for s in schemes:
        results[s] = runner.run(s, bps, keep_state=(s == "dynamic"))
    run_id = time.strftime("%Y%m%d_%H%M%S") + ("_FINAL" if final else "_dev")
    out = Path(outdir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    rf = cfg["backtest"]["rf_annual"]
    rows = []
    for s, r in results.items():
        m = perf(r.returns["net"], rf)
        m.update(name=s, avg_turnover=r.returns["turnover"].mean(), avg_n_held=r.returns["n_held"].mean(),
                 missing_exit=int(r.returns["missing_exit"].sum()))
        rows.append(m)
    metrics = pd.DataFrame(rows).set_index("name")
    # cost sensitivity for the headline strategy and equal-weight baseline
    sens = []
    for c in cfg["backtest"]["cost_grid_bps"]:
        for s in ["dynamic", "equal", "equal_weight"]:
            rr = runner.run(s, c)
            m = perf(rr.returns["net"], rf); m.update(strategy=s, cost_bps=c); sens.append(m)
    sens = pd.DataFrame(sens)
    boot = {}
    for base in ["equal_weight", "equal", "min_variance"]:
        boot[f"dynamic_vs_{base}"] = block_bootstrap_diff(results["dynamic"].returns["net"],
                                                         results[base].returns["net"],
                                                         cfg["backtest"]["bootstrap_block"],
                                                         cfg["backtest"]["bootstrap_reps"],
                                                         cfg["backtest"]["seed"], rf)
    metrics.to_csv(out / "metrics.csv"); sens.to_csv(out / "cost_sensitivity.csv")
    pd.DataFrame({s: r.returns["net"] for s, r in results.items()}).to_csv(out / "monthly_returns.csv")
    results["dynamic"].weights.to_csv(out / "weights_dynamic.csv")
    ap.ic.to_csv(out / "agent_ic.csv"); ap.regime_prob.to_csv(out / "regime_prob.csv")
    pd.DataFrame({a: ap.scores[a].iloc[-1] for a in ap.scores}).to_csv(out / "latest_scores.csv")
    json.dump({"bootstrap": boot, "folds": ap.fold_info, "git_commit": git_commit(), "final": final,
               "test_dates": [str(d.date()) for d in td]}, open(out / "run_info.json", "w"), indent=1, default=float)
    import yaml
    yaml.safe_dump(cfg, open(out / "config_used.yaml", "w"))
    from .explain.explain import explain_date, render_text
    from .explain.report import build_report
    last = list(results["dynamic"].state)[-1]
    ex = explain_date(runner, results["dynamic"], last, agents, cost_bps=bps)
    text = render_text(ex)
    json.dump(ex, open(out / "explanation_latest.json", "w"), indent=1, default=float)
    open(out / "explanation_latest.txt", "w").write(text)
    from .explain.writer import TemplateWriter
    open(out / "explanation_plain_english.txt", "w").write(TemplateWriter().write(ex))
    notes = ["Temporary EGX30 data (fixed 31-company list, survivorship bias, prices not verified as adjusted); results are provisional until Bloomberg data replaces it.",
             f"Costs are a PLACEHOLDER ({bps} bps one way) with a sensitivity grid; risk-free rate is a placeholder ({rf}); no EGX30 index benchmark yet.",
             "Fundamental, News and Macro agents are not built yet: momentum_rule and lowvol_rule are labeled stand-ins.",
             "Hyperparameters are defaults; the fully_invested choice was made after seeing development results (holdout untouched)." if not final else "FINAL holdout run.",
             f"Portfolio mode: {'fully invested' if cfg['portfolio'].get('fully_invested') else 'cash allowed'}."]
    build_report(out / "report.html", f"FinVerse AI testbed ({'FINAL holdout' if final else 'development'})",
                 metrics, sens, boot, results, ap, text, notes)
    from .state.store import StateStore
    from .state.recorder import record_run
    store = StateStore(out / "state.sqlite")
    record_run(store, run_id, ap, results["dynamic"])
    store.close()
    if final:
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(run_id)
    return out, results, ap, ds, runner, metrics, sens, boot


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default="config/default.yaml")
    r.add_argument("--final", action="store_true")
    r.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    out, results, ap, ds, runner, metrics, sens, boot = run(cfg, a.final, a.force)
    print(metrics[["months", "ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_turnover"]].round(3).to_string())
    print("results in", out)


if __name__ == "__main__":
    main()
