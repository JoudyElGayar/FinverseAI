# Decisions log
| # | Decision | Status | Notes |
|---|---|---|---|
| 1 | Hybrid: ML/rule agents score, LLM only explains | Agreed by team, supervisor sign-off pending | |
| 2 | Monthly rebalance, weekly as sensitivity | Default | |
| 3 | Start 2016 (2010 sensitivity); data ends 2025-10-12 | Default | confirm with supervisors |
| 4 | Final holdout 2024-01-01 onward, touched once | Default | |
| 5 | 2-state HMM, filtered probabilities | Pending supervisor sign-off | |
| 6 | Reliability = EW rank-IC on realized outcomes | Pending supervisor sign-off | |
| 7 | Long-only v1; futures hedge then shorting later | Agreed | short rules on EGX new, no history |
| 8 | Permission to reuse TradeXAI code granted | Confirmed by team | keep written record |
| 9 | Fundamental agent = transparent rule composite (ROE, ROA, net margin, NI growth, leverage, earnings yield), equal-weight ranks, fixed a priori | Built | too few annual observations for a fitted model; Bloomberg point-in-time data will replace the lag assumption |
| 10 | Revision study (scripts/revision_study.py): grid over data start, benchmark-relative optimizer, alpha_scale, softmax_k; selected on <=2021 only, reported on 2022-23 | Done | no setting selected on the tuning slice beat equal-weight on the validation slice; defaults unchanged; benchmark_relative kept as an option (default off) |
| 11 | Default config now fully_invested: true so the default run matches the reported results | Done | choice was made after seeing development results; holdout untouched |
| 12 | News agent: pre-trained ProsusAI/finbert scored once per article (no training), decayed mean over prior 30 days, ranked across stocks; only articles from earlier calendar days than the decision date count; stocks with no article are unscored | Built, disabled by default | source timezone undocumented, so same-day articles are excluded. Coverage is sparse and two companies (Fawry, Orascom Construction) have no news files. TradeXAI's trained sentiment weights are NOT reused: they were fit on one global 80/20 split. Scoring script not yet run (needs model download); agent untested on real data |
| 13 | Explanations: TemplateWriter produces plain English from logged numbers; LLMWriter is a pluggable wrapper that rejects any LLM output containing a number not in the template text | Built | LLM not connected (no API key); rule 5 enforced in code, tested |
| 14 | News agent evaluated on real scored articles (development period only): average rank-IC about -0.03; dynamic Sharpe 0.715 with it vs 0.728 without; News-only baseline Sharpe 0.654 with turnover above 1.0 | Done | no evidence it helps, so it stays disabled by default (agents.news.enabled: false). Two bugs found while integrating: company-name mismatch between sentiment file and dataset (fixed, tested) and engine crash when an agent scores no stock in a month (now holds, no trade) |
