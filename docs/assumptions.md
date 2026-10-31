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
