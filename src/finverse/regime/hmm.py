"""Two-state Gaussian HMM (1-d) with Baum-Welch and forward filtering. No external dependency."""
from __future__ import annotations
import numpy as np


def _norm_pdf(x, mu, var):
    return np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :]) / np.sqrt(2 * np.pi * var[None, :])


class GaussianHMM:
    def __init__(self, n_states=2, n_restarts=5, n_iter=60, seed=0):
        self.K, self.R, self.n_iter, self.seed = n_states, n_restarts, n_iter, seed
        self.pi = self.A = self.mu = self.var = None

    def _forward(self, B, pi, A):
        T, K = B.shape
        alpha = np.zeros((T, K)); c = np.zeros(T)
        a = pi * B[0]; c[0] = a.sum() + 1e-300; alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            c[t] = a.sum() + 1e-300
            alpha[t] = a / c[t]
        return alpha, c

    def _fit_once(self, x, rng):
        K = self.K
        v = np.var(x) + 1e-12
        mu = np.quantile(x, np.linspace(0.3, 0.7, K)) + rng.normal(0, np.sqrt(v) * 0.1, K)
        var = v * np.array([0.5 + rng.random() * 0.5 if k == 0 else 1.5 + rng.random() for k in range(K)])
        A = np.full((K, K), 0.1 / max(K - 1, 1)); np.fill_diagonal(A, 0.9)
        pi = np.full(K, 1.0 / K)
        ll_old = -np.inf
        for _ in range(self.n_iter):
            B = _norm_pdf(x, mu, var) + 1e-300
            alpha, c = self._forward(B, pi, A)
            T = len(x)
            beta = np.zeros((T, K)); beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = A * (alpha[:-1].T @ ((B[1:] * beta[1:]) / c[1:, None]))
            pi = gamma[0] / gamma[0].sum()
            A = xi / xi.sum(axis=1, keepdims=True)
            w = gamma.sum(axis=0)
            mu = (gamma * x[:, None]).sum(axis=0) / w
            var = (gamma * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / w
            var = np.maximum(var, 1e-8)
            ll = np.log(c).sum()
            if abs(ll - ll_old) < 1e-4 * max(1.0, abs(ll)) * 1e-2:
                break
            ll_old = ll
        return ll, (pi, A, mu, var)

    def fit(self, x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.R):
            ll, p = self._fit_once(x, rng)
            if best is None or ll > best[0]:
                best = (ll, p)
        pi, A, mu, var = best[1]
        order = np.argsort(var)          # state 0 = calmest, last state = most volatile
        self.pi, self.mu, self.var = pi[order], mu[order], var[order]
        self.A = A[np.ix_(order, order)]
        return self

    def filter(self, x):
        """Filtered state probabilities P(state_t | x_1..x_t). NaN observations carry the prior forward."""
        x = np.asarray(x, float)
        T, K = len(x), self.K
        out = np.zeros((T, K))
        prev = self.pi.copy()
        for t in range(T):
            pred = prev if t == 0 else prev @ self.A
            if np.isfinite(x[t]):
                b = _norm_pdf(np.array([x[t]]), self.mu, self.var)[0] + 1e-300
                post = pred * b
                post /= post.sum()
            else:
                post = pred
            out[t] = post
            prev = post
        return out
