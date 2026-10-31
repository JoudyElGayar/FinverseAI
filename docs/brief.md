# Brief
FinVerse AI: regime-adaptive, reliability-aware multi-agent system for explainable EGX portfolio decisions.
Supervisors: Dr. Sherif Aly (CSCE) and Dr. Wael (finance co-supervisor). Team of seven.

Agents (each outputs score 0-1, confidence, timestamp, source ids, drivers): Technical, Fundamental, Macro, News.
Coordinator weights agents by regime and recent reliability. Bull/Bear debate. cvxpy optimizer (long-only, cash,
position and turnover limits). Explanation layer with counterfactuals. Shared versioned state.

Testbed scope (this repo today): Technical ML agent + two clearly labeled stand-in rule agents, HMM regime,
reliability coordinator, optimizer, walk-forward backtest, numeric Bull/Bear + counterfactual explanations, HTML report.
Not yet: Fundamental, News, Macro agents (data/timestamp work pending), LLM text layer, UI, shorting.
