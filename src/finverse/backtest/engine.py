"""Walk-forward engine: fit agents per fold, collect out-of-sample scores, then build portfolios sequentially."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from ..regime.hmm import GaussianHMM
from ..coordinator.weights import agent_weights, combine
from ..optimizer.portfolio import optimize


@dataclass
class AgentPanel:
    dates: pd.DatetimeIndex
    scores: dict        # agent -> DataFrame (dates x stocks)
    confs: dict
    regime_prob: pd.DataFrame
    regime_hard: pd.Series
    ic: pd.DataFrame
    fold_info: list


@dataclass
class SchemeResult:
    name: str
    cost_bps: float
    returns: pd.DataFrame        # index dates: gross, cost, net, turnover, n_held, cash
    weights: pd.DataFrame        # dates x stocks (target)
    state: dict = field(default_factory=dict)


def select_test_dates(ds, cfg, final=False) -> pd.DatetimeIndex:
    b = cfg["backtest"]
    hold = pd.Timestamp(b["final_holdout_start"])
    r = ds.rebal
    ok = ds.label_end.reindex(r).notna()
    if final:
        m = (r >= hold) & ok
    else:
        m = (r.year >= b["first_test_year"]) & (r < hold) & ok & (ds.label_end.reindex(r) < hold)
    return r[m.values if hasattr(m, "values") else m]


def run_agents(ds, cfg, agents, test_dates) -> AgentPanel:
    rcfg = cfg["regime"]
    scores = {a.name: [] for a in agents}
    confs = {a.name: [] for a in agents}
    probs = []
    fold_info = []
    years = sorted(set(test_dates.year))
    for y in years:
        td = test_dates[test_dates.year == y]
        first = td[0]
        train = [d for d in ds.rebal if d >= ds.start and pd.notna(ds.label_end[d]) and ds.label_end[d] <= first]
        for a in agents:
            a.fit(ds, train)
        # regime: fit on market returns up to the fold start, filter forward through the fold year
        x_fit = ds.mkt_ret.loc[:first].dropna().values
        hmm = GaussianHMM(rcfg["n_states"], rcfg["n_restarts"], seed=cfg["backtest"]["seed"]).fit(x_fit)
        x_all = ds.mkt_ret.loc[:td[-1]]
        pr = pd.DataFrame(hmm.filter(x_all.values), index=x_all.index)
        probs.append(pr.reindex(td))
        fold_info.append({"year": int(y), "n_train_dates": len(train), "regime_mu": hmm.mu.tolist(),
                          "regime_var": hmm.var.tolist(), "regime_A": hmm.A.tolist()})
        for d in td:
            for a in agents:
                o = a.score(ds, d)
                scores[a.name].append(o.score.rename(d))
                confs[a.name].append(o.confidence.rename(d))
    sc = {k: pd.DataFrame(v) for k, v in scores.items()}
    cf = {k: pd.DataFrame(v) for k, v in confs.items()}
    rp = pd.concat(probs)
    rp.columns = [f"state{i}" for i in range(rp.shape[1])]
    hard = pd.Series(rp.values.argmax(axis=1), index=rp.index)
    ic = pd.DataFrame({k: [sc[k].loc[d].corr(ds.label.loc[d], method="spearman") for d in test_dates]
                       for k in sc}, index=test_dates)
    return AgentPanel(test_dates, sc, cf, rp, hard, ic, fold_info)


def _cov(ds, d, cols, window):
    r = ds.daily_ret.loc[:d].iloc[-window:][cols]
    r = r.fillna(0.0).values
    lw = LedoitWolf().fit(r)
    return lw.covariance_ * 21.0


class Runner:
    def __init__(self, cfg, ds, ap: AgentPanel):
        self.cfg, self.ds, self.ap = cfg, ds, ap
        self._cov_cache = {}
        self.agent_names = list(ap.scores)

    def cov(self, d, cols):
        key = (d, tuple(cols))
        if key not in self._cov_cache:
            self._cov_cache[key] = _cov(self.ds, d, list(cols), self.cfg["portfolio"]["cov_window_days"])
        return self._cov_cache[key]

    def run(self, scheme: str, cost_bps: float, keep_state=False) -> SchemeResult:
        ds, cfg, ap = self.ds, self.cfg, self.ap
        pcfg, ccfg = cfg["portfolio"], cfg["coordinator"]
        stocks = ds.stocks
        n = len(stocks)
        wdrift = pd.Series(0.0, index=stocks)
        rows, wrows, state = [], {}, {}
        first = True
        for k, d in enumerate(ap.dates):
            el = ds.elig.loc[d]
            info = {}
            if scheme in ("equal_weight", "buy_hold", "min_variance"):
                names = [s for s in stocks if el[s]]
                if scheme == "buy_hold" and not first:
                    target = wdrift.copy()
                elif scheme in ("equal_weight", "buy_hold"):
                    target = pd.Series(0.0, index=stocks); target[names] = 1.0 / len(names)
                else:
                    cv = self.cov(d, names)
                    p2 = dict(pcfg); p2["risk_aversion"] = 1.0
                    w, oi = optimize(np.zeros(len(names)), cv, wdrift[names].values, p2, cost_bps, first, full=True)
                    target = pd.Series(0.0, index=stocks); target[names] = w
                    info = oi
            else:
                if scheme.startswith("only:"):
                    a = scheme.split(":")[1]
                    comb = ap.scores[a].loc[d]
                    wts = {a: 1.0}
                else:
                    wts, info = agent_weights(self.agent_names, ap.ic, ap.regime_hard, ds.label_end.reindex(ap.dates), k, ccfg, scheme)
                    comb = combine({a: ap.scores[a].loc[d] for a in wts}, {a: ap.confs[a].loc[d] for a in wts}, wts)
                names = [s for s in stocks if el[s] and pd.notna(comb[s])]
                mu = pcfg["alpha_scale"] * (comb[names].values - 0.5)
                cv = self.cov(d, names)
                w, oi = optimize(mu, cv, wdrift[names].values, pcfg, cost_bps, first, full=bool(pcfg.get("fully_invested", False)))
                target = pd.Series(0.0, index=stocks); target[names] = w
                info = {**info, **oi}
                if keep_state:
                    state[d] = {"combined": comb, "agent_weights": wts, "info": info, "mu": pd.Series(mu, index=names),
                                "w_prev": wdrift.copy(), "regime_prob": ap.regime_prob.loc[d].to_dict(),
                                "regime": int(ap.regime_hard.loc[d])}
            R = ds.fwd_ret.loc[d].reindex(stocks)
            miss = int((R.isna() & (target > 0)).sum())
            R = R.fillna(0.0)
            turnover = float((target - wdrift).abs().sum())
            gross = float((target * R).sum())
            cost = turnover * cost_bps / 1e4
            net = gross - cost
            rows.append({"date": d, "gross": gross, "cost": cost, "net": net, "turnover": turnover,
                         "n_held": int((target > 1e-6).sum()), "cash": float(1 - target.sum()), "missing_exit": miss})
            wrows[d] = target
            wdrift = target * (1 + R) / (1 + gross)
            first = False
        ret = pd.DataFrame(rows).set_index("date")
        return SchemeResult(scheme, cost_bps, ret, pd.DataFrame(wrows).T, state)
