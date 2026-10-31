"""Self-contained HTML report (no external assets)."""
from __future__ import annotations
import base64, io, html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def build_report(path, title, metrics, sens, boot, results, ap, explanation_text, notes):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for s in ["dynamic", "equal", "equal_weight", "min_variance"]:
        if s in results:
            ((1 + results[s].returns["net"]).cumprod()).plot(ax=ax, label=s)
    ax.set_title("Net wealth (start = 1)"); ax.legend(); ax.grid(alpha=.3)
    img1 = _png(fig)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ap.regime_prob.iloc[:, -1].plot(ax=ax); ax.set_ylim(0, 1); ax.set_title("Filtered probability of the high-volatility regime")
    ax.grid(alpha=.3); img2 = _png(fig)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ap.ic.rolling(12, min_periods=6).mean().plot(ax=ax); ax.axhline(0, color="k", lw=.8)
    ax.set_title("Agent reliability: 12-month rolling rank-IC"); ax.grid(alpha=.3); img3 = _png(fig)
    m = metrics[["months", "ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_turnover", "avg_n_held"]]
    mt = m.to_html(float_format=lambda x: f"{x:.3f}", classes="t")
    st = sens.pivot(index="cost_bps", columns="strategy", values="sharpe").to_html(float_format=lambda x: f"{x:.2f}", classes="t")
    bt = "".join(f"<li>{html.escape(k)}: Sharpe difference {v['sharpe_diff']:+.2f}, 95% CI [{v['ci_low']:+.2f}, {v['ci_high']:+.2f}]</li>"
                 for k, v in boot.items() if v)
    nt = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    doc = f"""<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222}}
table.t{{border-collapse:collapse;font-size:13px}}table.t td,table.t th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
pre{{background:#f5f5f5;padding:12px;overflow:auto;font-size:12.5px}}.warn{{background:#fff4e5;border-left:4px solid #e69500;padding:8px 12px}}</style>
<h1>{html.escape(title)}</h1>
<div class=warn><b>Testbed results, provisional.</b><ul>{nt}</ul></div>
<h2>Performance (net of costs)</h2>{mt}<img src="{img1}" style="max-width:100%">
<h2>Sharpe vs cost assumption (bps one-way, placeholder grid)</h2>{st}
<h2>Dynamic vs baselines (block bootstrap)</h2><ul>{bt}</ul>
<h2>Regime</h2><img src="{img2}" style="max-width:100%">
<h2>Agent reliability</h2><img src="{img3}" style="max-width:100%">
<h2>Latest decision explained (numbers computed, not written by an LLM)</h2><pre>{html.escape(explanation_text)}</pre>"""
    open(path, "w").write(doc)
