"""Regressão logística condicional rápida (1 caso por estrato), por Newton-Raphson vetorizado.

Uso: b, se, ll = clogit(y, X, strata). Equivale a statsmodels ConditionalLogit para estratos 1:M,
mas usa np.add.reduceat (ordem de grandeza mais rápido em milhões de linhas).
Também oferece erro-padrão robusto por cluster (ex.: estação) via sanduíche.
"""
import numpy as np


def _prep(y, X, strata):
    o = np.argsort(strata, kind='stable')
    s = np.asarray(strata)[o]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    return o, starts


def clogit(y, X, strata, cluster=None, maxit=50, tol=1e-9, full=False):
    """Retorna (b, se, ll, se_cluster); com full=True retorna também a matriz de covariância robusta por cluster."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    o, st = _prep(y, X, strata)
    y, X = y[o], X[o]
    cl = None if cluster is None else np.asarray(cluster)[o]
    n, k = X.shape
    b = np.zeros(k)
    sid = np.repeat(np.arange(len(st)), np.diff(np.r_[st, n]))
    ll_old = -np.inf
    for it in range(maxit):
        eta = X @ b
        mx = np.maximum.reduceat(eta, st)
        w = np.exp(eta - mx[sid])
        sw = np.add.reduceat(w, st)
        p = w / sw[sid]
        ll = np.sum(y * eta) - np.sum(mx + np.log(sw))
        xbar = np.add.reduceat(p[:, None] * X, st)          # E[x] por estrato
        xc = X - xbar[sid]
        score_i = (y[:, None] - p[:, None]) * X             # soma por estrato = x_caso - E[x]
        g = score_i.sum(0)
        H = (p[:, None] * xc).T @ xc
        step = np.linalg.solve(H, g)
        b = b + step
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
    se_cl = None; cov_cl = None
    if cl is not None:
        U = np.add.reduceat(score_i, st)                     # escore por estrato
        cs = cl[st]
        uc, inv = np.unique(cs, return_inverse=True)
        Uc = np.zeros((len(uc), k)); np.add.at(Uc, inv, U)
        G = len(uc)
        meat = Uc.T @ Uc * G / (G - 1)
        cov_cl = cov @ meat @ cov
        se_cl = np.sqrt(np.diag(cov_cl))
    if full:
        return b, se, ll, se_cl, cov_cl
    return b, se, ll, se_cl


if __name__ == '__main__':
    # teste contra statsmodels
    from statsmodels.discrete.conditional_models import ConditionalLogit
    rng = np.random.default_rng(1)
    S = 3000; m = 4
    strata = np.repeat(np.arange(S), m)
    X = rng.normal(size=(S * m, 2))
    eta = X @ np.array([0.5, -0.3])
    y = np.zeros(S * m)
    for s in range(S):
        e = np.exp(eta[s*m:(s+1)*m]); y[s*m + rng.choice(m, p=e/e.sum())] = 1
    b, se, ll, _ = clogit(y, X, strata)
    r = ConditionalLogit(y, X, groups=strata).fit(disp=0)
    print(b, r.params, se, r.bse)
    assert np.allclose(b, r.params, atol=1e-4) and np.allclose(se, r.bse, atol=1e-4)
    print('ok')
