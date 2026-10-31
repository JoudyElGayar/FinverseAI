# Data rules
- Source (temporary): TradeXAI EGX30 prices, cleaned and causally imputed by our pipeline. See audit log shipped with the data.
- Imputed rows carry `imputed=True`. Stocks mostly imputed recently are ineligible; imputed rows never count toward history.
- Universe is a fixed list of 31 companies in the data -> survivorship bias. Bloomberg point-in-time membership replaces it.
- Prices are not verified as corporate-action adjusted (12 flagged jumps unresolved). Bloomberg adjusted prices replace them.
- No risk-free rate or index series yet: Sharpe uses rf from config (PLACEHOLDER) and there is no EGX30 benchmark yet.
- Fundamentals and news need publication-date checks before any use; macro series not available yet.
