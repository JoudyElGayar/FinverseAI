# Architecture
```
prices/volume -> eligibility -> features -> agents (score, confidence, drivers)
                                   |             |
market returns -> HMM (filtered) --+-> coordinator (reliability x regime) -> combined score
                                                       |
                         trailing covariance ----------+-> optimizer -> target weights -> costs -> simulated returns
                                                       |
                                          explanation layer (Bull/Bear numeric, counterfactuals)
```
Timing: signal at close of day t, trade at close of t+1 (config `exec_lag_days`).
Label for date r: return of each stock from the trade date of r to the trade date of the next rebalance,
converted to a cross-sectional rank percentile (0-1). A label is "realized" only at its end date.

Walk-forward: expanding window, refit each January, training rows must have label end <= first test decision date (purge).
Reliability: exponentially weighted rank correlation of past scores vs realized labels (only realized ones).
Weights: softmax(k * reliability) with blend of overall and same-regime reliability; equal weights during burn-in.
Regime: 2-state Gaussian HMM on daily equal-weight market return, parameters fitted on data up to the fold start,
forward-filtered probabilities only (no smoothing).
LLM role: explanation text only, from logged numbers (not implemented yet).
