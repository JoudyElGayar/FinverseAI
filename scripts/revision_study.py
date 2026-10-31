"""Revision study. Settings are SELECTED on the tuning slice only (<= 2021-12-31) and REPORTED on the validation slice
(>= 2022-01-01). The final holdout (2024 onward) is never touched. Run: python scripts/revision_study.py"""
import itertools, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from finverse.config import load_config, override
from finverse.data.dataset import build_dataset
from finverse.cli import build_agents
from finverse.backtest.engine import select_test_dates, run_agents, Runner
from finverse.backtest.metrics import perf, block_bootstrap_diff

SPLIT = pd.Timestamp("2021-12-31")
base = load_config("config/default.yaml")
base = override(base, **{"portfolio.fully_invested": True})
out = []
panels = {}
for start in ["2016-01-01", "2010-01-01"]:
    cfg = override(base, **{"data.start": start})
    ds = build_dataset(cfg)
    td = select_test_dates(ds, cfg)
    panels[start] = (cfg, ds, run_agents(ds, cfg, build_agents(cfg), td))
    print("agents done", start, len(td), flush=True)

def sharpe(x):
    return perf(x)["sharpe"] if len(x) > 3 else np.nan

for start, rel, alpha, k in itertools.product(["2016-01-01", "2010-01-01"], [False, True], [0.05, 0.10, 0.20], [5.0, 20.0]):
    cfg, ds, ap = panels[start]
    c2 = override(cfg, **{"portfolio.benchmark_relative": rel, "portfolio.alpha_scale": alpha, "coordinator.softmax_k": k})
    r = Runner(c2, ds, ap)
    x = r.run("dynamic", 25.0).returns["net"]
    out.append(dict(start=start[:4], bench_rel=rel, alpha_scale=alpha, softmax_k=k,
                    sharpe_tune=sharpe(x[x.index <= SPLIT]), sharpe_val=sharpe(x[x.index > SPLIT]),
                    ret_val=perf(x[x.index > SPLIT])["ann_return"], dd_val=perf(x[x.index > SPLIT])["max_drawdown"],
                    sharpe_all=sharpe(x)))
    print(out[-1], flush=True)
res = pd.DataFrame(out)
# baselines per start (same dates)
bl = {}
for start, (cfg, ds, ap) in panels.items():
    r = Runner(cfg, ds, ap)
    for s in ["equal_weight", "min_variance"]:
        x = r.run(s, 25.0).returns["net"]
        bl[(start[:4], s)] = dict(sharpe_tune=sharpe(x[x.index <= SPLIT]), sharpe_val=sharpe(x[x.index > SPLIT]),
                                  ret_val=perf(x[x.index > SPLIT])["ann_return"], dd_val=perf(x[x.index > SPLIT])["max_drawdown"])
res.to_csv("results/revision_grid.csv", index=False)
pd.DataFrame(bl).T.to_csv("results/revision_baselines.csv")
print(res.sort_values("sharpe_tune", ascending=False).head(8).round(3).to_string())
print(pd.DataFrame(bl).T.round(3))
