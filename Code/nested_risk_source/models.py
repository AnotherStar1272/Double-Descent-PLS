"""Numerical estimators and deterministic equivalents used by the paper.

The routines avoid scikit-learn so that a PLS component is exactly the
Krylov/CG component defined in the manuscript. Risks returned here are excess
prediction risks; the independent test-noise variance is added by the
experiment layer when the figure calls for full prediction error.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Iterable

import numpy as np
from scipy import linalg, optimize

Array = np.ndarray


@dataclass(frozen=True)
class PLSPath:
    estimates: dict[int, Array]
    effective_k: dict[int, int]
    basis: Array


def _rank_threshold(diagonal: Array, shape: tuple[int, int]) -> float:
    if diagonal.size == 0:
        return 0.0
    return max(shape) * np.finfo(float).eps * float(np.max(np.abs(diagonal)))


def minnorm_ols(X: Array, y: Array) -> Array:
    """Moore--Penrose least-squares solution, using QR when full rank."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("X must be two-dimensional and y must match its rows")
    n, p = X.shape
    if n >= p:
        Q, R, piv = linalg.qr(X, mode="economic", pivoting=True, check_finite=False)
        diagonal = np.diag(R)
        if np.count_nonzero(np.abs(diagonal) > _rank_threshold(diagonal, X.shape)) == p:
            z = linalg.solve_triangular(R, Q.T @ y, check_finite=False)
            beta = np.empty(p)
            beta[piv] = z
            return beta
    else:
        Q, R = linalg.qr(X.T, mode="economic", check_finite=False)
        diagonal = np.diag(R)
        if np.count_nonzero(np.abs(diagonal) > _rank_threshold(diagonal, X.shape)) == n:
            return Q @ linalg.solve_triangular(R.T, y, lower=True, check_finite=False)
    return linalg.lstsq(X, y, cond=None, lapack_driver="gelsd", check_finite=False)[0]


def pls_krylov_path(
    X: Array, y: Array, k_values: Iterable[int], tol: float = 1e-12
) -> PLSPath:
    """Orthogonalized Krylov PLS path using two-pass modified Gram--Schmidt."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    requested = sorted({int(k) for k in k_values})
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("X must be two-dimensional and y must match its rows")
    if not requested or requested[0] < 1:
        raise ValueError("k_values must contain positive integers")
    n, p = X.shape
    v = X.T @ y / n
    threshold = tol * max(1.0, float(np.linalg.norm(v)))
    q_columns: list[Array] = []
    t_columns: list[Array] = []
    for _ in range(requested[-1]):
        for _pass in range(2):
            for q in q_columns:
                v -= q * float(q @ v)
        norm_v = float(np.linalg.norm(v))
        if not np.isfinite(norm_v) or norm_v <= threshold:
            break
        q = v / norm_v
        t = X @ q
        q_columns.append(q)
        t_columns.append(t)
        v = X.T @ t / n
    Q = np.column_stack(q_columns) if q_columns else np.empty((p, 0))
    T = np.column_stack(t_columns) if t_columns else np.empty((n, 0))
    estimates: dict[int, Array] = {}
    effective_k: dict[int, int] = {}
    for k in requested:
        k_eff = min(k, Q.shape[1])
        effective_k[k] = k_eff
        if k_eff == 0:
            estimates[k] = np.zeros(p)
        else:
            theta = linalg.lstsq(
                T[:, :k_eff], y, cond=None, lapack_driver="gelsy", check_finite=False
            )[0]
            estimates[k] = Q[:, :k_eff] @ theta
    return PLSPath(estimates, effective_k, Q)


def pls_krylov(X: Array, y: Array, k: int, tol: float = 1e-12) -> Array:
    return pls_krylov_path(X, y, [k], tol).estimates[int(k)]


def pls1_closed_form(X: Array, y: Array, tol: float = 1e-14) -> Array:
    n = X.shape[0]
    s = X.T @ y / n
    xs = X @ s
    denominator = float(xs @ xs) / n
    if denominator <= tol * max(1.0, float(s @ s)):
        return np.zeros(X.shape[1])
    return (float(s @ s) / denominator) * s


def exact_prediction_risk(beta_hat: Array, beta: Array, eigvals: Array | None = None) -> float:
    difference = np.asarray(beta_hat) - np.asarray(beta)
    if eigvals is None:
        return float(difference @ difference)
    return float(np.sum(np.asarray(eigvals) * difference**2))


def mp_moment(order: int, c: float) -> float:
    """Moment of the p-by-p Marchenko--Pastur ESD (zero atom included)."""
    if order < 0:
        raise ValueError("order must be nonnegative")
    if order == 0:
        return 1.0
    return float(sum(
        comb(order, j) * comb(order, j - 1) * c ** (j - 1) / order
        for j in range(1, order + 1)
    ))


@lru_cache(maxsize=8)
def _legendre_rule(size: int) -> tuple[Array, Array]:
    return np.polynomial.legendre.leggauss(size)


@lru_cache(maxsize=512)
def mp_quadrature(c: float, size: int = 500) -> tuple[Array, Array, float]:
    """Quadrature nodes/weights for MP(c), plus its mass at zero."""
    if c <= 0:
        raise ValueError("c must be positive")
    x, w = _legendre_rule(size)
    theta = (x + 1.0) * np.pi / 4.0
    a, b = (1.0 - np.sqrt(c)) ** 2, (1.0 + np.sqrt(c)) ** 2
    sine, cosine = np.sin(theta), np.cos(theta)
    lam = a + (b - a) * sine**2
    jac = (b - a) * 2.0 * sine * cosine * np.pi / 4.0
    density = np.sqrt(np.maximum((b-lam)*(lam-a), 0.0)) / (2*np.pi*c*lam)
    weights = w * density * jac
    atom = max(0.0, 1.0 - 1.0/c)
    weights *= (1.0 - atom) / weights.sum()
    return lam, weights, atom


def _cg_diagonal_filters(lam: Array, rhs: Array, k_max: int) -> list[Array]:
    """q_k(lam) such that CG's x_k=q_k(diag(lam)) rhs."""
    if k_max < 1:
        raise ValueError("k_max must be positive")
    x = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual.copy()
    rr = float(residual @ residual)
    filters: list[Array] = []
    for _ in range(k_max):
        Ad = lam * direction
        denominator = float(direction @ Ad)
        if denominator <= np.finfo(float).tiny:
            current = np.divide(x, rhs, out=np.zeros_like(x), where=rhs != 0)
            filters.extend([current.copy() for _ in range(k_max-len(filters))])
            break
        alpha = rr / denominator
        x += alpha * direction
        residual -= alpha * Ad
        rr_new = float(residual @ residual)
        filters.append(np.divide(x, rhs, out=np.zeros_like(x), where=rhs != 0))
        if rr_new <= 1e-28 * max(1.0, float(rhs @ rhs)):
            filters.extend([filters[-1].copy() for _ in range(k_max-len(filters))])
            break
        direction = residual + (rr_new/rr) * direction
        rr = rr_new
    return filters


def gradient_flow_spectral_factors(
    singular: Array, times: Array, n_samples: int
) -> Array:
    """Stable spectral factors for gradient flow from the zero initializer.

    The returned ``(rank, n_times)`` array contains
    ``(1-exp(-s**2*t/n))/s``.  ``expm1`` avoids cancellation for small
    singular values, and the continuous value at ``s=0`` is zero.
    """
    singular = np.asarray(singular, dtype=float)
    times = np.asarray(times, dtype=float)
    if singular.ndim != 1 or times.ndim != 1:
        raise ValueError("singular and times must be one-dimensional")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if np.any(singular < 0) or np.any(times < 0):
        raise ValueError("singular values and times must be nonnegative")
    numerator = -np.expm1(-np.outer(singular**2 / n_samples, times))
    return np.divide(
        numerator,
        singular[:, None],
        out=np.zeros_like(numerator),
        where=singular[:, None] != 0,
    )


def isotropic_pls_theory(
    c: float, r2: float, sigma2: float, k_max: int, quadrature_size: int = 500
) -> list[dict[str, float | int]]:
    """Stable arbitrary-depth deterministic equivalents via MP quadrature.

    A diagonal CG representation avoids the ill-conditioned raw-moment Hankel
    solve at depths such as k=20 while evaluating the same limit.
    """
    lam, weight, atom = mp_quadrature(float(c), quadrature_size)
    omega = weight * (r2*lam**2 + sigma2*c*lam)
    filters = _cg_diagonal_filters(lam, np.sqrt(omega), k_max)
    rows: list[dict[str, float | int]] = []
    for k, q in enumerate(filters, 1):
        residual = 1.0 - lam*q
        risk = r2 * (float(np.sum(weight*residual**2)) + atom)
        risk += sigma2*c*float(np.sum(weight*lam*q**2))
        rows.append({
            "k": k,
            "risk": risk,
            "train_mse": r2 + sigma2 - float(np.sum(omega*q)),
        })
    return rows


def ols_asymptotic_risk(c: float | Array, r2: float, sigma2: float) -> Array:
    c_array = np.asarray(c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            c_array < 1.0,
            sigma2*c_array/(1.0-c_array),
            np.where(c_array > 1.0,
                     r2*(1.0-1.0/c_array) + sigma2/(c_array-1.0), np.inf),
        )


def pls1_candidate_risk(c: float | Array, r2: float, sigma2: float) -> Array:
    c_array = np.asarray(c, dtype=float)
    A = r2*(1.0+c_array) + sigma2*c_array
    B = r2*(1.0+3.0*c_array+c_array**2) + sigma2*c_array*(1.0+c_array)
    a = A/B
    return r2 - 2.0*a*r2 + a*a*A


def two_eigen_pls1_risk(
    c: float | Array, r2: float, sigma2: float, values=(4.0, 1.0)
) -> Array:
    """General-covariance PLS1 excess risk for H=G, two equal atoms."""
    c = np.asarray(c, dtype=float)
    vals = np.asarray(values)
    mu1, mu2, mu3 = (np.mean(vals**j) for j in (1, 2, 3))
    nu1, nu2, nu3 = mu1, mu2, mu3
    A = r2*(nu2+c*mu1*nu1) + sigma2*c*mu1
    B = r2*(nu3+2*c*mu1*nu2+c*mu2*nu1+c**2*mu1**2*nu1)
    B += sigma2*c*(mu2+c*mu1**2)
    D = r2*(nu3+c*mu2*nu1) + sigma2*c*mu2
    alpha = A/B
    return r2*nu1 - 2*r2*nu2*alpha + D*alpha**2


def two_eigen_ols_risk(
    c: float | Array, r2: float, sigma2: float, values=(4.0, 1.0)
) -> Array:
    """Correlated minimum-norm OLS excess-risk limit in the manuscript."""
    result = []
    h = np.asarray(values, dtype=float)
    for ratio in np.atleast_1d(np.asarray(c, dtype=float)):
        if ratio < 1.0:
            value = sigma2*ratio/(1.0-ratio)
        elif ratio == 1.0:
            value = np.inf
        else:
            target = 1.0 - 1.0/ratio
            fun = lambda eta: float(np.mean(1.0/(1.0+eta*ratio*h))-target)
            upper = 1.0
            while fun(upper) > 0:
                upper *= 2.0
            eta = optimize.brentq(fun, 0.0, upper)
            den = (1.0+eta*ratio*h)**2
            moment_ratio = np.mean(h**2/den)/np.mean(h/den)
            bias = r2*(1.0+ratio*eta*moment_ratio)*np.mean(h/den)
            variance = sigma2*ratio*eta*moment_ratio
            value = bias + variance
        result.append(value)
    out = np.asarray(result)
    return out if np.ndim(c) else out[0]


def pure_noise_pls_risk(c: float | Array, k: int, sigma2: float = 1.0) -> Array:
    """Exact Chebyshev pure-noise PLS excess risk."""
    ratios = np.atleast_1d(np.asarray(c, dtype=float))
    answer = []
    for ratio in ratios:
        x = (1.0+ratio)/(2.0*np.sqrt(ratio))
        sequence = [1.0, 2.0*x]
        for _j in range(2, k+1):
            sequence.append(2.0*x*sequence[-1]-sequence[-2])
        answer.append(sigma2*float(np.sum(np.square(sequence[:k])))/sequence[k]**2)
    out = np.asarray(answer)
    return out if np.ndim(c) else out[0]


def pure_noise_train_error(c: float | Array, k: int, sigma2: float = 1.0) -> Array:
    ratios = np.atleast_1d(np.asarray(c, dtype=float))
    answer = []
    for ratio in ratios:
        x = (1.0+ratio)/(2.0*np.sqrt(ratio))
        sequence = [1.0, 2.0*x]
        for _j in range(2, k+1):
            sequence.append(2.0*x*sequence[-1]-sequence[-2])
        answer.append(sigma2*(1.0-np.sqrt(ratio)*sequence[k-1]/sequence[k]))
    out = np.asarray(answer)
    return out if np.ndim(c) else out[0]


def boundary_phi(z: float | Array) -> Array:
    z = np.asarray(z, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = (np.sinh(2*z)-2*z)/(4*z*np.sinh(z)**2)
    return np.where(np.abs(z) < 1e-6, 1.0/3.0-2.0*z**2/45.0, value)


def critical_gradient_flow(
    t: Array, r2: float = 1.0, sigma2: float = 1.0, quadrature_size: int = 1200
) -> tuple[Array, Array]:
    lam, weight, atom = mp_quadrature(1.0, quadrature_size)
    times = np.asarray(t, dtype=float)[:, None]
    decay = np.exp(-times*lam[None, :])
    bias = r2*(np.sum(weight[None, :]*decay**2, axis=1)+atom)
    variance = sigma2*np.sum(weight[None, :]*(1.0-decay)**2/lam[None, :], axis=1)
    return bias, variance


def population_pls_path(
    eigvals: Array, beta: Array, k_values: Iterable[int]
) -> dict[int, Array]:
    """Population Krylov projections in the Sigma inner product."""
    eigvals = np.asarray(eigvals, dtype=float)
    beta = np.asarray(beta, dtype=float)
    requested = sorted(set(map(int, k_values)))
    v = eigvals*beta
    columns: list[Array] = []
    output: dict[int, Array] = {}
    for step in range(1, requested[-1]+1):
        for _pass in range(2):
            for q in columns:
                v -= q*float(q@v)
        norm = float(np.linalg.norm(v))
        if norm > 1e-13:
            q = v/norm
            columns.append(q)
            v = eigvals*q
        if step in requested:
            if not columns:
                output[step] = np.zeros_like(beta)
            else:
                Q = np.column_stack(columns)
                gram = Q.T@(eigvals[:, None]*Q)
                rhs = Q.T@(eigvals*beta)
                output[step] = Q@np.linalg.lstsq(gram, rhs, rcond=None)[0]
    last = np.zeros_like(beta)
    for k in requested:
        if k in output:
            last = output[k]
        else:
            output[k] = last.copy()
    return output
