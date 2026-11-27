"""Finite-atom deterministic equivalents for general-covariance PLS.

The implementation mirrors the rooted noncrossing formulas in the paper.  It
is deliberately separate from the simulation runners so that numerical plots
can compare an empirical PLS path with a theory calculation that never sees
the simulated sample covariance matrix.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np

from wick_check import evaluate_terms, noncrossing_partitions, rooted_signature


@lru_cache(maxsize=None)
def _rooted_terms(
    order: int, mark_position: int | None = None
) -> Counter[tuple[int, int, tuple[int, ...]]]:
    """Cached rooted monomials for ``NC(order)``.

    ``mark_position`` is zero based.  Thus the marked position for
    :math:`E_{a,b}` is ``a``, since the signal root is position zero and the
    inserted population covariance is attached to feature position ``a+1``
    in the paper's one-based notation.
    """
    if order < 1:
        raise ValueError("order must be positive")
    if mark_position is not None and not 0 <= mark_position < order:
        raise ValueError("mark_position must lie in range(order)")
    return Counter(
        rooted_signature(partition, mark_position)
        for partition in noncrossing_partitions(order)
    )


def _probability_weights(weights: np.ndarray, name: str) -> np.ndarray:
    if weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if np.any(weights < 0):
        raise ValueError(f"{name} must be nonnegative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive total mass")
    return weights / total


def _atomic_moments(
    values: np.ndarray, weights: np.ndarray, maximum_order: int
) -> dict[int, float]:
    moments = {0: 1.0}
    power = np.ones_like(values)
    for order in range(1, maximum_order + 1):
        power = power * values
        moments[order] = float(weights @ power)
    return moments


def moment_functionals(
    c: float,
    values,
    h_weights,
    g_weights,
    k_max: int = 3,
) -> dict[str, Any]:
    """Evaluate ``M``, ``N``, ``C``, ``D`` and ``E`` for finite atomic laws.

    Parameters
    ----------
    c:
        Positive limiting aspect ratio ``p/n``.
    values:
        Common finite support for the population spectral law ``H`` and the
        signal-alignment law ``G``.  Zero weights allow the two laws to have
        different effective supports.
    h_weights, g_weights:
        Nonnegative atomic masses.  They are normalized to sum to one.
    k_max:
        Largest PLS depth for which the returned moments are sufficient.

    Returns
    -------
    A dictionary whose scalar moment families are keyed by their integer
    orders and whose ``E`` family is keyed by ``(a, b)``.
    """
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and positive")
    if not isinstance(k_max, (int, np.integer)) or k_max < 1:
        raise ValueError("k_max must be a positive integer")

    atoms = np.asarray(values, dtype=float)
    h_mass = _probability_weights(np.asarray(h_weights, dtype=float), "h_weights")
    g_mass = _probability_weights(np.asarray(g_weights, dtype=float), "g_weights")
    if atoms.ndim != 1 or atoms.size == 0 or not np.all(np.isfinite(atoms)):
        raise ValueError("values must be a nonempty finite one-dimensional array")
    if h_mass.shape != atoms.shape or g_mass.shape != atoms.shape:
        raise ValueError("values, h_weights and g_weights must have the same length")
    if np.any(atoms < 0):
        raise ValueError("population eigenvalues must be nonnegative")

    # H_k needs N through order 2k+1; marked moments need powers through
    # order 2k+1 as well.  Computing all four one-root families on this
    # common range keeps the public result easy to inspect.
    maximum_order = 2 * int(k_max) + 1
    mu = _atomic_moments(atoms, h_mass, maximum_order + 1)
    nu = _atomic_moments(atoms, g_mass, maximum_order + 1)

    M: dict[int, float] = {0: 1.0}
    N: dict[int, float] = {0: 1.0}
    C: dict[int, float] = {0: nu[1]}
    D: dict[int, float] = {0: mu[1]}
    for order in range(1, maximum_order + 1):
        terms = _rooted_terms(order)
        marked_root_terms = _rooted_terms(order, 0)
        M[order] = evaluate_terms(terms, c, mu, mu)
        N[order] = evaluate_terms(terms, c, mu, nu)
        C[order] = evaluate_terms(marked_root_terms, c, mu, nu)
        D[order] = evaluate_terms(marked_root_terms, c, mu, mu)

    E: dict[tuple[int, int], float] = {}
    for a in range(1, int(k_max) + 1):
        for b in range(1, int(k_max) + 1):
            order = a + b
            E[a, b] = evaluate_terms(_rooted_terms(order, a), c, mu, nu)

    return {
        "mu": mu,
        "nu": nu,
        "M": M,
        "N": N,
        "C": C,
        "D": D,
        "E": E,
    }


def _preconditioned_hankel_solve(
    hankel: np.ndarray, rhs: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve a positive Hankel system after diagonal equilibration."""
    hankel = (np.asarray(hankel, dtype=float) + np.asarray(hankel, dtype=float).T) / 2.0
    rhs = np.asarray(rhs, dtype=float)
    diagonal = np.diag(hankel)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise np.linalg.LinAlgError("limiting Hankel matrix has a nonpositive diagonal")

    scale = np.sqrt(diagonal)
    equilibrated = hankel / np.outer(scale, scale)
    equilibrated = (equilibrated + equilibrated.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(equilibrated)
    tolerance = 128 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(equilibrated, 2)))
    if eigenvalues[0] <= tolerance:
        raise np.linalg.LinAlgError(
            "limiting Krylov Hankel matrix is singular or numerically noncoercive"
        )

    scaled_solution = np.linalg.solve(equilibrated, rhs / scale)
    solution = scaled_solution / scale
    residual = hankel @ solution - rhs
    diagnostics = {
        "hankel_condition": float(np.linalg.cond(hankel)),
        "hankel_condition_scaled": float(np.linalg.cond(equilibrated)),
        "hankel_scaled_min_eigenvalue": float(eigenvalues[0]),
        "hankel_relative_residual": float(
            np.linalg.norm(residual) / max(1.0, np.linalg.norm(rhs))
        ),
    }
    return solution, diagnostics


def general_theory(
    c: float,
    values,
    h_weights,
    g_weights,
    r2: float,
    sigma2: float,
    k_max: int = 3,
) -> list[dict[str, Any]]:
    """Return fixed-depth general-covariance PLS limits through ``k_max``.

    Each row includes excess prediction risk, training MSE, the limiting
    Krylov system, and numerical diagnostics.  Prediction *test* error is
    obtained by adding ``sigma2`` to ``risk``.
    """
    if not np.isfinite(r2) or r2 < 0:
        raise ValueError("r2 must be finite and nonnegative")
    if not np.isfinite(sigma2) or sigma2 < 0:
        raise ValueError("sigma2 must be finite and nonnegative")

    moments = moment_functionals(c, values, h_weights, g_weights, k_max)
    M, N = moments["M"], moments["N"]
    C, D, E = moments["C"], moments["D"], moments["E"]
    nu1 = moments["nu"][1]

    mbar = {
        order: r2 * N[order + 2] + sigma2 * c * M[order + 1]
        for order in range(2 * int(k_max))
    }

    rows: list[dict[str, Any]] = []
    for k in range(1, int(k_max) + 1):
        hankel = np.array(
            [[mbar[a + b + 1] for b in range(k)] for a in range(k)],
            dtype=float,
        )
        hbar = np.array([mbar[a] for a in range(k)], dtype=float)
        b_vector = np.array([r2 * C[a + 1] for a in range(k)], dtype=float)
        test_gram = np.array(
            [
                [r2 * E[a + 1, b + 1] + sigma2 * c * D[a + b + 1]
                 for b in range(k)]
                for a in range(k)
            ],
            dtype=float,
        )

        symmetry_error = float(np.max(np.abs(test_gram - test_gram.T)))
        test_gram = (test_gram + test_gram.T) / 2.0
        theta, diagnostics = _preconditioned_hankel_solve(hankel, hbar)
        risk = float(r2 * nu1 - 2.0 * theta @ b_vector + theta @ test_gram @ theta)
        train_mse = float(r2 * nu1 + sigma2 - hbar @ theta)
        gram_eigenvalues = np.linalg.eigvalsh(test_gram)

        rows.append({
            "k": k,
            "risk": risk,
            "train_mse": train_mse,
            "full_test_error": risk + sigma2,
            "theta": theta,
            "Hbar": hankel,
            "hbar": hbar,
            "b": b_vector,
            "A": test_gram,
            "test_gram_symmetry_error": symmetry_error,
            "test_gram_min_eigenvalue": float(gram_eigenvalues[0]),
            **diagnostics,
        })
    return rows


__all__ = ["general_theory", "moment_functionals"]
