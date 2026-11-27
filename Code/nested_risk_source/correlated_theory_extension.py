"""Reproducible PLS1/2/5 theory extension for the correlated main figure.

The simulation data are read only.  The deterministic equivalents are
evaluated from rooted/marked noncrossing moments, independently of those data.
An 80-decimal solve is used for the released curves, with a float64 solve
reported separately to expose high-order Hankel conditioning.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, product
from math import comb
from pathlib import Path

import mpmath as mp
import numpy as np

from general_theory import _preconditioned_hankel_solve, general_theory
from models import isotropic_pls_theory, two_eigen_ols_risk, two_eigen_pls1_risk
from wick_check import noncrossing_partitions, rooted_signature


@lru_cache(maxsize=None)
def recursive_noncrossing(size: int) -> tuple:
    """Construct NC(size) directly using the block containing position zero.

    Once that block is chosen, each open interval between successive block
    members (and the final tail) is an independent noncrossing partition.
    This visits the Catalan objects only, not all Bell-number set partitions.
    """
    if size < 0:
        raise ValueError("size must be nonnegative")
    if size == 0:
        return ((),)
    answer = []
    for root_size in range(1, size + 1):
        for remaining in combinations(range(1, size), root_size - 1):
            root = (0,) + remaining
            boundaries = root + (size,)
            intervals = []
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                offset = left + 1
                intervals.append(tuple(
                    tuple(tuple(index + offset for index in block) for block in partition)
                    for partition in recursive_noncrossing(right - offset)
                ))
            for pieces in product(*intervals):
                blocks = (root,) + tuple(block for piece in pieces for block in piece)
                answer.append(tuple(sorted(blocks, key=lambda block: block[0])))
    expected = comb(2 * size, size) // (size + 1)
    if len(answer) != expected:
        raise AssertionError(f"NC({size}) count is not Catalan")
    return tuple(answer)


@lru_cache(maxsize=None)
def fast_terms(order: int, mark: int | None = None) -> Counter:
    if order < 1 or (mark is not None and not 0 <= mark < order):
        raise ValueError("invalid rooted/marked order")
    return Counter(rooted_signature(partition, mark)
                   for partition in recursive_noncrossing(order))


def _number(value, high_precision: bool):
    return mp.mpf(str(value)) if high_precision else float(value)


def _evaluate(terms, c, mu, nu):
    # All summands are nonnegative; cancellation occurs only in the risk
    # quadratic form, which is precisely why the released value uses mp.
    return sum(coefficient * c ** c_power * nu[root_power]
               * _multiply(mu[power] for power in trace_powers)
               for (c_power, root_power, trace_powers), coefficient in terms.items())


def _multiply(values):
    answer = 1
    for value in values:
        answer *= value
    return answer


def extended_moments(c, values, h_weights, g_weights, k_max=5, high_precision=False):
    cast = lambda value: _number(value, high_precision)
    c = cast(c)
    atoms = [cast(value) for value in values]
    h_mass, g_mass = [cast(x) for x in h_weights], [cast(x) for x in g_weights]
    if c <= 0 or k_max < 1 or not atoms or any(value < 0 for value in atoms):
        raise ValueError("invalid aspect ratio, depth or eigenvalues")
    if len(atoms) != len(h_mass) or len(atoms) != len(g_mass):
        raise ValueError("atomic support and masses must have matching lengths")
    if min(h_mass + g_mass) < 0 or sum(h_mass) <= 0 or sum(g_mass) <= 0:
        raise ValueError("invalid spectral masses")
    h_mass = [x / sum(h_mass) for x in h_mass]
    g_mass = [x / sum(g_mass) for x in g_mass]
    maximum = 2 * k_max + 1
    mu = {q: sum(w * x ** q for w, x in zip(h_mass, atoms))
          for q in range(maximum + 2)}
    nu = {q: sum(w * x ** q for w, x in zip(g_mass, atoms))
          for q in range(maximum + 2)}
    moments = {"mu": mu, "nu": nu, "M": {0: cast(1)}, "N": {0: cast(1)},
               "C": {0: nu[1]}, "D": {0: mu[1]}, "E": {}}
    for q in range(1, maximum + 1):
        moments["M"][q] = _evaluate(fast_terms(q), c, mu, mu)
        moments["N"][q] = _evaluate(fast_terms(q), c, mu, nu)
        moments["C"][q] = _evaluate(fast_terms(q, 0), c, mu, nu)
        moments["D"][q] = _evaluate(fast_terms(q, 0), c, mu, mu)
    for a in range(1, k_max + 1):
        for b in range(1, k_max + 1):
            moments["E"][a, b] = _evaluate(fast_terms(a + b, a), c, mu, nu)
    return moments


def extended_theory(c, values=(4, 1), h_weights=(.5, .5), g_weights=(.5, .5),
                    r2=5, sigma2=1, k_max=5, dps=80):
    """Return high-precision risks and float64 stability diagnostics."""
    if dps < 30:
        raise ValueError("at least 30 decimal digits are required")
    with mp.workdps(dps):
        moments_mp = extended_moments(c, values, h_weights, g_weights, k_max, True)
        moments_fp = extended_moments(c, values, h_weights, g_weights, k_max, False)
        answer = []
        for k in range(1, k_max + 1):
            versions = []
            for high_precision, moments in ((False, moments_fp), (True, moments_mp)):
                cast = lambda value: _number(value, high_precision)
                ratio, signal, noise = cast(c), cast(r2), cast(sigma2)
                M, N, C, D, E = [moments[key] for key in ("M", "N", "C", "D", "E")]
                m = {q: signal * N[q + 2] + noise * ratio * M[q + 1]
                     for q in range(2 * k)}
                H = [[m[a + b + 1] for b in range(k)] for a in range(k)]
                h = [m[a] for a in range(k)]
                b_vector = [signal * C[a + 1] for a in range(k)]
                A = [[signal * E[a + 1, b + 1] + noise * ratio * D[a + b + 1]
                      for b in range(k)] for a in range(k)]
                if high_precision:
                    H, h, b_vector, A = map(mp.matrix, (H, h, b_vector, A))
                    theta = mp.lu_solve(H, h)
                    risk = signal * moments["nu"][1] - 2 * (theta.T * b_vector)[0]
                    risk += (theta.T * A * theta)[0]
                    train = signal * moments["nu"][1] + noise - (h.T * theta)[0]
                    residual = mp.norm(H * theta - h) / max(mp.mpf(1), mp.norm(h))
                    versions.append({"risk": float(risk), "train_mse": float(train),
                                     "risk_decimal": mp.nstr(risk, 60),
                                     "train_decimal": mp.nstr(train, 60),
                                     "mp_relative_residual": float(residual)})
                else:
                    H, h, b_vector, A = map(np.asarray, (H, h, b_vector, A))
                    theta, diagnostics = _preconditioned_hankel_solve(H, h)
                    risk = signal * moments["nu"][1] - 2 * theta @ b_vector + theta @ A @ theta
                    train = signal * moments["nu"][1] + noise - h @ theta
                    versions.append({"risk_float64": float(risk), "train_float64": float(train),
                                     "test_gram_symmetry_error": float(np.max(np.abs(A - A.T))),
                                     "test_gram_min_eigenvalue": float(np.linalg.eigvalsh((A + A.T) / 2)[0]),
                                     **diagnostics})
            row = {"c": float(c), "k": k, **versions[0], **versions[1]}
            row["risk_float64_abs_difference"] = abs(row["risk_float64"] - row["risk"])
            row["risk_float64_relative_difference"] = row["risk_float64_abs_difference"] / max(abs(row["risk"]), 1e-15)
            row["train_float64_abs_difference"] = abs(row["train_float64"] - row["train_mse"])
            answer.append(row)
        return answer


def sha_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def structural_checks() -> dict:
    levels = []
    for q in range(1, 12):
        fast = recursive_noncrossing(q)
        level = {"order": q, "count": len(fast), "catalan": comb(2 * q, q) // (q + 1),
                 "unique": len(set(fast)) == len(fast)}
        if q <= 7:
            old = noncrossing_partitions(q)
            level["matches_independent_bell_enumeration"] = set(fast) == set(old)
            level["marked_coefficients_match"] = all(
                fast_terms(q, mark) == Counter(rooted_signature(p, mark) for p in old)
                for mark in (None, *range(q)))
        levels.append(level)
    return {"levels": levels, "ok": all(
        row["count"] == row["catalan"] and row["unique"]
        and row.get("matches_independent_bell_enumeration", True)
        and row.get("marked_coefficients_match", True) for row in levels)}
