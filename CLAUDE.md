# Project rules for Claude Code

FinVerse AI thesis testbed. Read `docs/` first (brief, architecture, decisions, data_rules, assumptions).

## Non-negotiable rules
1. **No assumptions presented as facts.** Do not write exact numbers in unverified statements. Config values
   marked PLACEHOLDER must be reported as placeholders, with a sensitivity grid.
2. **No leakage.** Anything used at decision date t may use only data available at close of t. Any new feature,
   agent or weighting rule needs a case in `tests/test_leakage.py`; the future-poisoning test must pass.
3. **Final holdout (2024-01-01 onward) is touched once** via `--final`. Never tune on it.
4. **Same costs for every strategy and baseline.**
5. LLMs only write explanations/debate text from logged numeric scores. They never produce a score, a weight or a trade.
6. Keep the pipeline repeatable: fixed seeds, config-driven, results written to `results/<run_id>/` with the config and git commit.
7. Log every non-obvious decision in `docs/decisions.md`.

## Commands
- `pytest -q` run tests (CI runs the same)
- `python -m finverse.cli run --config config/default.yaml` development run (test years before the holdout)
- `python -m finverse.cli run --config config/default.yaml --final` holdout run (once)

## Data
`data/` is gitignored. Cleaned files live in `data/clean_imputed/*.csv` (columns: Date, Price, Open, High, Low, Volume, imputed, ...).
