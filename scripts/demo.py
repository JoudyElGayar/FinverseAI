"""One-command demo: development backtest, then print the headline table, the plain-English explanation of the
latest decision, and where the report is. Never touches the final holdout.
    python scripts/demo.py [--config config/default.yaml]"""
import argparse, sys
sys.path.insert(0, "src")
from finverse.config import load_config
from finverse.cli import run


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/default.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out, results, _, _, _, metrics, _, boot = run(cfg, final=False, force=False)
    print("\n=== Development period results (net of costs; cost level is a PLACEHOLDER) ===")
    print(metrics[["months", "ann_return", "ann_vol", "sharpe", "max_drawdown"]].round(3).to_string())
    print("\n=== Sharpe difference vs baselines (block-bootstrap CI) ===")
    for k, v in boot.items():
        print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()})
    print("\n=== Latest decision, in plain English ===")
    print((out / "explanation_plain_english.txt").read_text())
    print("\nOpen this in a browser:", out / "report.html")


if __name__ == "__main__":
    main()
