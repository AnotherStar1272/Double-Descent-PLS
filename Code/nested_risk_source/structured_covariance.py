"""Population models and independent proportional limits for covariance experiments.

Simulation uses exact finite-p eigenvalues and Gaussian rotation invariance.
Theory uses limiting H,G, not sample eigenvalues or fitted simulation curves.
"""
from __future__ import annotations

from functools import lru_cache
from math import comb

import mpmath as mp
import numpy as np
from scipy import linalg, optimize

from correlated_theory_extension import fast_terms, _evaluate
from general_theory import _preconditioned_hankel_solve

COVARIANCES = ("ar0", "ar03", "ar06", "ar07", "ar09", "banded", "block_ar", "spike")
ALIGNMENTS = ("weak", "balanced", "strong")
DEPTHS = (1, 2, 3, 5)
RHOS = dict(ar0=0., ar03=.3, ar06=.6, ar07=.7, ar09=.9)
ALPHAS = dict(weak=-1, balanced=0, strong=1)
SPIKE_MASSES = dict(weak=0., balanced=.5, strong=1.)


@lru_cache(maxsize=512)
def ar_eigenvalues(p, rho):
    if rho == 0:
        return np.ones(p)
    if p == 1:
        return np.ones(1)
    diagonal = np.full(p, 1 + rho*rho)
    diagonal[[0, -1]] = 1
    inverse_eigenvalues = linalg.eigvalsh_tridiagonal(diagonal, np.full(p-1, -rho))
    return (1-rho*rho)/inverse_eigenvalues[::-1]


@lru_cache(maxsize=512)
def population(covariance, p, alignment):
    if covariance in RHOS:
        eig = ar_eigenvalues(p, RHOS[covariance]).copy()
    elif covariance == "banded":
        eig = np.sort(1 + .8*np.cos(np.pi*np.arange(1, p+1)/(p+1)))
    elif covariance == "block_ar":
        eig = np.sort(np.r_[ar_eigenvalues(p//2, .3), ar_eigenvalues(p-p//2, .8)])
    elif covariance == "spike":
        eig = np.r_[np.ones(p-1), 9.]
    else:
        raise ValueError(covariance)
    if covariance == "spike":
        mass = SPIKE_MASSES[alignment]
        weights = np.r_[np.full(p-1, (1-mass)/(p-1)), mass]
    else:
        weights = eig**ALPHAS[alignment]
        weights /= weights.sum()
    # This coefficient has exactly unit population signal variance.
    beta = np.sqrt(weights / (eig @ weights))
    eig.setflags(write=False)
    beta.setflags(write=False)
    return eig, beta


def ar_moment(q, rho):
    rho = mp.mpf(str(rho))
    x = (1+rho*rho)/(1-rho*rho)
    if q == -1:
        return x
    if q == 0:
        return mp.mpf(1)
    # Integral of [(1-rho^2)/(1-2rho cos(w)+rho^2)]^q.
    return mp.legendre(q-1, x)


def spectral_moment(covariance, q):
    if covariance in RHOS:
        return ar_moment(q, RHOS[covariance])
    if covariance == "block_ar":
        return (ar_moment(q, .3)+ar_moment(q, .8))/2
    if covariance == "spike":
        return mp.mpf(1)
    if covariance == "banded":
        a = mp.mpf(".4")
        if q == -1:
            return 1/mp.sqrt(1-4*a*a)
        return sum(mp.mpf(comb(q, 2*j)*comb(2*j, j))*a**(2*j)
                   for j in range(q//2+1))
    raise ValueError(covariance)


def population_moments(covariance, alignment, maximum=12):
    mu = {q: spectral_moment(covariance, q) for q in range(maximum+2)}
    if covariance == "spike":
        mass = mp.mpf(str(SPIKE_MASSES[alignment]))
        nu = {q: (1-mass)+mass*mp.mpf(9)**q for q in range(maximum+2)}
    else:
        alpha = ALPHAS[alignment]
        normalizer = spectral_moment(covariance, alpha)
        nu = {q: spectral_moment(covariance, q+alpha)/normalizer
              for q in range(maximum+2)}
    return mu, nu


def quadrature(covariance, alignment, size=4096):
    w = 2*np.pi*(np.arange(size)+.5)/size
    def ar(rho):
        return (1-rho*rho)/(1-2*rho*np.cos(w)+rho*rho)
    if covariance in RHOS:
        values = ar(RHOS[covariance])
    elif covariance == "banded":
        values = 1+.8*np.cos(w)
    elif covariance == "block_ar":
        values = np.r_[ar(.3), ar(.8)]
    elif covariance == "spike":
        mass = SPIKE_MASSES[alignment]
        return np.array([1., 9.]), np.array([1., 0.]), np.array([1-mass, mass])
    else:
        raise ValueError(covariance)
    h = np.full(values.size, 1/values.size)
    g = h*values**ALPHAS[alignment]
    return values, h, g/g.sum()


def ols_limit(gamma, covariance, alignment, snr, sigma2=1.):
    if gamma == 1:
        return float("inf")
    if gamma < 1:
        return sigma2*gamma/(1-gamma)
    t, h, g = quadrature(covariance, alignment)
    r2 = snr*sigma2/float(g @ t)
    # z = gamma*eta in the manuscript; solve monotonically in log(z).
    target = 1-1/gamma
    z = np.exp(optimize.brentq(lambda logz: h @ (1/(1+np.exp(logz)*t))-target,
                              -40, 40, xtol=1e-13))
    denom = (1+z*t)**2
    ratio = (h @ (t*t/denom))/(h @ (t/denom))
    return float(r2*(1+z*ratio)*(g @ (t/denom))+sigma2*z*ratio)


def theory(gamma, covariance, alignment, snrs=(1., 5.), dps=80):
    """Fixed depths 1,2,3,5, with MP arithmetic and separate float64 checks."""
    with mp.workdps(dps):
        mu, nu = population_moments(covariance, alignment)
        c = mp.mpf(str(gamma))
        M, N, C, D, E = {}, {}, {}, {}, {}
        for q in range(1, 12):
            M[q] = _evaluate(fast_terms(q), c, mu, mu)
            N[q] = _evaluate(fast_terms(q), c, mu, nu)
            C[q] = _evaluate(fast_terms(q, 0), c, mu, nu)
            D[q] = _evaluate(fast_terms(q, 0), c, mu, mu)
        for a in range(1, 6):
            for b in range(1, 6):
                E[a,b] = _evaluate(fast_terms(a+b, a), c, mu, nu)
        rows = []
        for snr in snrs:
            r2 = mp.mpf(str(snr))/nu[1]
            m = {j: r2*N[j+2]+c*M[j+1] for j in range(10)}
            for k in DEPTHS:
                H = mp.matrix([[m[a+b+1] for b in range(k)] for a in range(k)])
                h = mp.matrix([m[a] for a in range(k)])
                bvec = mp.matrix([r2*C[a+1] for a in range(k)])
                A = mp.matrix([[r2*E[a+1,b+1]+c*D[a+b+1] for b in range(k)] for a in range(k)])
                theta = mp.lu_solve(H, h)
                value = r2*nu[1]-2*(theta.T*bvec)[0]+(theta.T*A*theta)[0]
                train = r2*nu[1]+1-(h.T*theta)[0]
                hf, hv, bv, af = [np.asarray(x.tolist(), dtype=float) for x in (H,h,bvec,A)]
                hv, bv = hv.ravel(), bv.ravel()
                tf, diagnostic = _preconditioned_hankel_solve(hf,hv)
                fp = float(snr-2*tf@bv+tf@af@tf)
                rows.append(dict(covariance=covariance, alignment=alignment, gamma=gamma,
                                 snr=snr, method=f"PLS{k}", k=k, theory=float(value),
                                 train_theory=float(train), r2_limit=float(r2),
                                 float64_relative_difference=abs(fp/float(value)-1),
                                 mp_relative_residual=float(mp.norm(H*theta-h)/max(1,mp.norm(h))),
                                 **diagnostic))
            rows.append(dict(covariance=covariance, alignment=alignment, gamma=gamma,
                             snr=snr, method="OLS", k=0,
                             theory=ols_limit(gamma,covariance,alignment,snr),
                             train_theory=max(1-gamma,0), r2_limit=float(r2)))
        return rows


def ols_multiple(X, Y):
    """Reuse one stable QR factorization for paired response configurations."""
    n,p = X.shape
    if n >= p:
        Q,R,piv = linalg.qr(X, mode="economic", pivoting=True, check_finite=False)
        z = linalg.solve_triangular(R,Q.T@Y,check_finite=False)
        result = np.empty((p,Y.shape[1]))
        result[piv] = z
        return result
    Q,R = linalg.qr(X.T,mode="economic",check_finite=False)
    return Q@linalg.solve_triangular(R.T,Y,lower=True,check_finite=False)
