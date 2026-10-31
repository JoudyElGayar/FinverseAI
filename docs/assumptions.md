# Assumptions and verification status
| Item | Status |
|---|---|
| Prices adjusted for splits/dividends | NOT verified |
| Universe point-in-time | NO (fixed list, survivorship bias) |
| Costs (config) | PLACEHOLDER, confirm with broker |
| Risk-free rate | PLACEHOLDER (0), needs T-bill data |
| Execution at next close at observed price | Assumption; no slippage model beyond cost_bps |
| EGX liquidity/capacity for real money | NOT assessed |
| Short selling | Out of scope (v1 long-only) |
| GPU / Ubuntu 26.04 toolchain | NOT verified (nvidia-smi driver mismatch reported) |
| Hyperparameters (softmax_k, halflife, alpha_scale, risk_aversion) | Defaults; tune on development folds only |
| Fundamentals publication dates | NOT in source files; modelled with a lag (PLACEHOLDER 9 months, grid 6/9/12) |
| Fundamentals data | Annual only, no announcement dates; ratio tables are a current snapshot (not usable historically); analyst fair values are dated but almost vanish after 2022 |
| Fundamentals coverage | 29 of 31 companies have statements (EKHOA and Madinet Nasr missing) |
| EPS vs price consistency | NOT verified (splits/adjustments could distort earnings yield) |
