"""Paired PLS/OLS simulations at 36 equally log-spaced dimension ratios.

Raw samples and provenance go to outputs_snr_comparison, never theory/.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from models import _cg_diagonal_filters, minnorm_ols, pls_krylov_path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Experiment/outputs_snr_comparison"
SNRS = (1.0, 2.33, 3.66, 5.0)
COUNTS = (1, 2, 5, 10)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def grid(n):
    nominal = np.geomspace(.1, 10, 36)
    dimensions = np.rint(n*nominal).astype(int)
    assert len(np.unique(dimensions)) == 36 and dimensions.min() >= 10
    return nominal, dimensions


def spectral_risks(X, noise):
    """Exact sample PLS and minimum-norm OLS errors in right-singular coordinates.

    Only the smaller Gram matrix is diagonalized. The orthogonal signal
    component for p>n is included explicitly. No limiting coefficients are used.
    """
    n, p = X.shape
    roots = np.sqrt(SNRS)
    if p <= n:
        lam, U = eigh(X.T@X/n, check_finite=False)
        beta = U[0, :, None]*roots
        rhs = lam[:, None]*beta + (U.T@(X.T@noise/n))[:, None]
        null_error = np.zeros(len(SNRS))
    else:
        lam, U = eigh(X@X.T/n, check_finite=False)
        signal = U.T@X[:, 0]
        beta = signal[:, None]/np.sqrt(n*lam[:, None])*roots
        response = signal[:, None]*roots + (U.T@noise)[:, None]
        rhs = np.sqrt(lam[:, None]/n)*response
        null_error = np.maximum(0, np.array(SNRS)-np.sum(beta**2, axis=0))
    if np.min(lam) <= 0:
        raise ArithmeticError("Nonpositive eigenvalue in the smaller Gaussian Gram matrix")
    risks = np.empty((len(SNRS), len(COUNTS)+1))
    for index in range(len(SNRS)):
        filters = _cg_diagonal_filters(lam, rhs[:, index], max(COUNTS))
        for column, k in enumerate(COUNTS):
            estimate = filters[k-1]*rhs[:, index]
            risks[index, column] = np.sum((estimate-beta[:, index])**2)+null_error[index]
        risks[index, -1] = np.sum((rhs[:, index]/lam-beta[:, index])**2)+null_error[index]
    return risks


def validate_estimator():
    rng = np.random.default_rng(92801)
    discrepancy = 0.0
    for n, p in ((30, 15), (30, 30), (30, 55)):
        X, noise = rng.normal(size=(n, p)), rng.normal(size=n)
        spectral = spectral_risks(X, noise)
        for index, snr in enumerate(SNRS):
            beta = np.zeros(p)
            beta[0] = np.sqrt(snr)
            y = X@beta+noise
            path = pls_krylov_path(X, y, COUNTS)
            actual = [np.sum((path.estimates[k]-beta)**2) for k in COUNTS]
            actual.append(np.sum((minnorm_ols(X, y)-beta)**2))
            difference = np.max(np.abs(spectral[index]-actual)/np.maximum(1, actual))
            discrepancy = max(discrepancy, float(difference))
    assert discrepancy < 1e-8
    return discrepancy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    raw_path = DATA/"raw.csv"
    if raw_path.exists():
        raise FileExistsError("Simulation already exists; use the plotting script to redraw")
    error = validate_estimator()
    nominal, dimensions = grid(args.n)
    rows = []
    started = time.perf_counter()
    for number, (target, p) in enumerate(zip(nominal, dimensions), 1):
        for rep in range(args.repeats):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, args.n, int(p), rep]))
            X = rng.normal(size=(args.n, int(p)))
            noise = rng.normal(size=args.n)
            risks = spectral_risks(X, noise)
            for index, snr in enumerate(SNRS):
                for column, k in enumerate((*COUNTS, 0)):
                    rows.append((args.n, int(p), target, p/args.n, snr, k, rep, risks[index, column]))
        print(f"[{number}/36] p={p}, gamma={p/args.n:g}: {args.repeats} paired samples", flush=True)
    raw = pd.DataFrame(rows, columns=["n", "p", "nominal_gamma", "gamma", "snr", "k", "replicate", "risk"])
    assert len(raw) == 36*args.repeats*4*5
    assert np.isfinite(raw.risk).all() and raw.risk.ge(0).all()
    assert not raw.duplicated(["p", "snr", "k", "replicate"]).any()
    summary = raw.groupby(["n", "p", "nominal_gamma", "gamma", "snr", "k"], as_index=False).agg(
        mean_risk=("risk", "mean"), sd_risk=("risk", "std"), repeats=("risk", "size"))
    summary["mcse"] = summary.sd_risk/np.sqrt(summary.repeats)
    assert len(summary) == 36*4*5 and summary.repeats.eq(args.repeats).all()
    raw.to_csv(raw_path, index=False)
    summary.to_csv(DATA/"summary.csv", index=False)
    report = dict(passed=True, n=args.n, repeats=args.repeats, seed=args.seed,
                  sigma2=1, snrs=list(SNRS), components=list(COUNTS), ols_code=0,
                  grid="36 log-equally-spaced targets in [0.1,10]; p=round(n*gamma); plot actual p/n",
                  nominal_gamma=nominal.tolist(), p_values=dimensions.tolist(),
                  features="iid N(0,1)", errors="iid N(0,1) independent of X",
                  coefficient="sqrt(SNR)*e_1; deterministic",
                  pairing="same X and noise for all SNRs and estimators in each repetition",
                  risk="exact squared coefficient error, Sigma=I",
                  independent_training_samples=36*args.repeats, raw_rows=len(raw),
                  max_independent_fit_relative_difference=error,
                  elapsed_seconds=time.perf_counter()-started,
                  files_sha256={p.name:digest(p) for p in (raw_path, DATA/"summary.csv")},
                  source_sha256={p.name:digest(p) for p in (Path(__file__), Path(__file__).with_name("models.py"))})
    (DATA/"simulation_manifest.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("passed", "independent_training_samples", "raw_rows",
                     "elapsed_seconds", "max_independent_fit_relative_difference")}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
