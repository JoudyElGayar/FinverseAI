# FinVerse AI

Regime-adaptive, reliability-aware multi-agent system for explainable EGX portfolio decisions (senior thesis, AUC).

**Status: testbed.** Runs end to end on temporary cleaned EGX30 data. Results are provisional until Bloomberg
point-in-time data replaces it. See `docs/assumptions.md` for what is and is not verified.

## Quick start
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
# put the cleaned data in data/clean_imputed/ (not in Git)
pytest -q
python -m finverse.cli run --config config/default.yaml
```
Outputs go to `results/<run_id>/` (metrics, returns, weights, explanations, `report.html`).

## Layout
- `src/finverse/data` loader, eligibility, labels, features
- `src/finverse/agents` Technical ML agent + stand-in rule agents
- `src/finverse/regime` 2-state HMM (filtered probabilities only)
- `src/finverse/coordinator` reliability + regime-aware weighting
- `src/finverse/optimizer` long-only portfolio optimizer (cvxpy, scipy fallback)
- `src/finverse/backtest` walk-forward engine, costs, metrics, baselines
- `src/finverse/explain` numeric Bull/Bear, counterfactuals, report
- `tests/` leakage tests (a future-poisoning test must always pass)
- `docs/` brief, architecture, decisions log, data rules, assumptions
