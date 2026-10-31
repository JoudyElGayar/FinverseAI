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
