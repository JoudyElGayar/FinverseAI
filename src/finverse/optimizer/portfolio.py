"""Long-only portfolio optimizer with cash, position and turnover limits.

maximize  mu'w - gamma * w'Sigma w - cost * ||w - w_prev||_1
s.t.      w >= 0, w <= max_position, sum(w) + cash = 1, cash >= min_cash, ||w - w_prev||_1 <= turnover_cap

Backends: cvxpy (primary, used when installed) and scipy SLSQP (fallback, tested here).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

try:  # pragma: no cover - depends on environment
    import cvxpy as cp
    HAVE_CVXPY = True
except Exception:  # pragma: no cover
    cp = None
    HAVE_CVXPY = False


def _solve_cvxpy(mu, cov, w_prev, gamma, cost, max_pos, min_cash, cap, full=False):
    n = len(mu)
    w = cp.Variable(n)
    turn = cp.norm1(w - w_prev)
    obj = mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(cov)) - cost * turn
    cons = [w >= 0, w <= max_pos, (cp.sum(w) == 1 - min_cash) if full else (cp.sum(w) <= 1 - min_cash)]
    if cap is not None:
        cons.append(turn <= cap)
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.CLARABEL) if "CLARABEL" in cp.installed_solvers() else prob.solve()
    if w.value is None:
        return None
    return np.clip(np.asarray(w.value).ravel(), 0, None)


def _solve_scipy(mu, cov, w_prev, gamma, cost, max_pos, min_cash, cap, full=False):
    n = len(mu)
    # x = [w, p, m]; w - w_prev = p - m; turnover = sum(p + m)
    def f(x):
        w, p, m = x[:n], x[n:2 * n], x[2 * n:]
        return -(mu @ w - gamma * w @ cov @ w - cost * (p.sum() + m.sum()))

    def g(x):
        w = x[:n]
        gr = np.zeros(3 * n)
        gr[:n] = -(mu - 2 * gamma * cov @ w)
        gr[n:] = cost
        return gr

    cons = [{"type": "eq", "fun": lambda x: x[:n] - w_prev - x[n:2 * n] + x[2 * n:],
             "jac": lambda x: np.hstack([np.eye(n), -np.eye(n), np.eye(n)])},
            ({"type": "eq", "fun": lambda x: (1 - min_cash) - x[:n].sum(),
              "jac": lambda x: np.hstack([-np.ones(n), np.zeros(2 * n)])} if full else
             {"type": "ineq", "fun": lambda x: (1 - min_cash) - x[:n].sum(),
              "jac": lambda x: np.hstack([-np.ones(n), np.zeros(2 * n)])})]
    if cap is not None:
        cons.append({"type": "ineq", "fun": lambda x: cap - x[n:].sum(),
                     "jac": lambda x: np.hstack([np.zeros(n), -np.ones(2 * n)])})
    bnds = [(0, max_pos)] * n + [(0, None)] * (2 * n)
    x0 = np.hstack([np.clip(w_prev, 0, max_pos), np.zeros(2 * n)])
    d = x0[:n] - w_prev
    x0[n:2 * n] = np.clip(d, 0, None); x0[2 * n:] = np.clip(-d, 0, None)
    res = minimize(f, x0, jac=g, bounds=bnds, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-10})
    if not res.success and res.status not in (0, 9):
        return None
    w = np.clip(res.x[:n], 0, max_pos)
    return w


def optimize(mu, cov, w_prev, pcfg: dict, cost_bps: float = 0.0, first_build=False, backend=None, full=False):
    """Return (weights ndarray, info dict). Weights exclude cash (cash = 1 - sum)."""
    mu = np.asarray(mu, float); cov = np.asarray(cov, float); w_prev = np.asarray(w_prev, float)
    n = len(mu)
    backend = backend or pcfg.get("backend", "auto")
    if backend == "auto":
        backend = "cvxpy" if HAVE_CVXPY else "scipy"
    gamma = pcfg["risk_aversion"]; max_pos = pcfg["max_position"]; min_cash = pcfg["min_cash"]
    cap = None if first_build else pcfg.get("turnover_cap")
    cost = cost_bps / 1e4
    if full:
        max_pos = max(max_pos, 1.01 * (1 - min_cash) / n)
    # keep max_pos feasible: need n*max_pos >= 1 - min_cash is NOT required (cash can absorb the rest)
    cov = 0.5 * (cov + cov.T) + 1e-10 * np.eye(n)
    fn = _solve_cvxpy if backend == "cvxpy" else _solve_scipy
    relax = 1.0
    for attempt in range(5):
        c = None if cap is None else cap * relax
        w = fn(mu, cov, w_prev, gamma, cost, max_pos, min_cash, c, full)
        if w is not None and w.sum() <= 1 - min_cash + 1e-6:
            break
        relax *= 1.5
        w = None
    if w is None:
        w = np.clip(w_prev, 0, max_pos)
        if w.sum() > 1 - min_cash:
            w = w / w.sum() * (1 - min_cash)
        return w, {"backend": backend, "status": "fallback_keep_previous", "relax": relax}
    w = np.where(w < 1e-6, 0.0, w)
    return w, {"backend": backend, "status": "ok", "relax": relax}
