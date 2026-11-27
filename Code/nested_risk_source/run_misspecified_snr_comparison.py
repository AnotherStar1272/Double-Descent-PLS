"""Two omitted-feature PLS/OLS comparisons, with audited paired Monte Carlo.

Run once; use --plot-only to redraw without changing the simulation data.
Figures: Experiment/theory and Paper/Figs (PNG only).
Data and reproducibility checks: Experiment/outputs_misspecified_comparison.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from hashlib import sha256
from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits

from models import _cg_diagonal_filters, minnorm_ols, ols_asymptotic_risk, pls1_candidate_risk, pls_krylov_path
from plot_pls_snr_comparison import high_precision_risk, risk_path
from run_pls_snr_monte_carlo import grid

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Experiment/outputs_misspecified_comparison"
FIGURES = ROOT / "Experiment/theory"
COUNTS = (1, 2, 5, 10)
EXPONENTS = (2, 5)
TOTAL = 5.0
SIGMA2 = 1.0
SOURCES = ("run_misspecified_snr_comparison.py", "models.py",
           "plot_pls_snr_comparison.py", "run_pls_snr_monte_carlo.py")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parameters(gamma, exponent):
    missing = TOTAL * (1 + gamma) ** (-exponent)
    return TOTAL - missing, SIGMA2 + missing, missing


def limiting_path(gamma, exponent, nodes=64):
    signal, noise, missing = parameters(gamma, exponent)
    return missing + noise * risk_path(gamma, signal / noise, nodes)


def ols_limit(gamma, exponent):
    signal, noise, missing = parameters(gamma, exponent)
    return missing + float(ols_asymptotic_risk(gamma, signal, noise))


def sample_errors(X, omitted, noise):
    """Exact sample errors, including omitted signal and the p>n null space."""
    n, p = X.shape
    signal, _, missing = parameters(p / n, np.array(EXPONENTS))
    roots = np.sqrt(signal)
    errors = omitted[:, None] * np.sqrt(missing) + noise[:, None]
    if p <= n:
        lam, U = eigh(X.T @ X / n, check_finite=False)
        beta = U[0, :, None] * roots
        rhs = lam[:, None] * beta + U.T @ (X.T @ errors / n)
        null_error = np.zeros(len(EXPONENTS))
    else:
        lam, U = eigh(X @ X.T / n, check_finite=False)
        projected_signal = U.T @ X[:, 0]
        beta = projected_signal[:, None] / np.sqrt(n * lam[:, None]) * roots
        response = projected_signal[:, None] * roots + U.T @ errors
        rhs = np.sqrt(lam[:, None] / n) * response
        null_error = signal - np.sum(beta ** 2, axis=0)
        assert np.min(null_error) > -1e-9
        null_error = np.maximum(0, null_error)
    if np.min(lam) <= 0:
        raise ArithmeticError("Nonpositive eigenvalue of the smaller Gaussian Gram matrix")
    result = np.empty((len(EXPONENTS), len(COUNTS) + 1))
    for index in range(len(EXPONENTS)):
        filters = _cg_diagonal_filters(lam, rhs[:, index], max(COUNTS))
        for column, k in enumerate(COUNTS):
            estimate = filters[k - 1] * rhs[:, index]
            result[index, column] = (np.sum((estimate - beta[:, index]) ** 2)
                                     + null_error[index] + missing[index])
        result[index, -1] = (np.sum((rhs[:, index] / lam - beta[:, index]) ** 2)
                             + null_error[index] + missing[index])
    return result


def validate_sample_errors():
    rng = np.random.default_rng(928202)
    maximum = 0.0
    for n, p in ((30, 15), (30, 30), (30, 55)):
        X = rng.normal(size=(n, p))
        omitted, noise = rng.normal(size=(2, n))
        spectral = sample_errors(X, omitted, noise)
        for index, a in enumerate(EXPONENTS):
            signal, _, missing = parameters(p / n, a)
            beta = np.zeros(p)
            beta[0] = np.sqrt(signal)
            y = X @ beta + np.sqrt(missing) * omitted + noise
            direct = pls_krylov_path(X, y, COUNTS)
            expected = [missing + np.sum((direct.estimates[k] - beta) ** 2) for k in COUNTS]
            expected.append(missing + np.sum((minnorm_ols(X, y) - beta) ** 2))
            maximum = max(maximum, float(np.max(np.abs(spectral[index] - expected)
                                               / np.maximum(1, expected))))
    assert maximum < 1e-8
    return maximum


def simulate(n, repeats, seed):
    if (DATA / "raw.csv").exists():
        raise FileExistsError("Existing simulation preserved: use --plot-only")
    fit_error = validate_sample_errors()
    nominal, dimensions = grid(n)
    rows = []
    started = time.perf_counter()
    for number, (target, p) in enumerate(zip(nominal, dimensions), 1):
        for rep in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, n, int(p), rep]))
            X = rng.normal(size=(n, int(p)))
            omitted, noise = rng.normal(size=(2, n))
            values = sample_errors(X, omitted, noise)
            for index, a in enumerate(EXPONENTS):
                for column, k in enumerate((*COUNTS, 0)):
                    rows.append((n, int(p), target, p / n, a, k, rep, values[index, column]))
        print(f"[{number}/36] p={p}, gamma={p/n:g}: {repeats} paired samples", flush=True)
    raw = pd.DataFrame(rows, columns=["n", "p", "nominal_gamma", "gamma", "a", "k", "replicate", "error"])
    assert len(raw) == 36 * repeats * len(EXPONENTS) * (len(COUNTS) + 1)
    assert np.isfinite(raw.error).all() and raw.error.ge(0).all()
    assert not raw.duplicated(["p", "a", "k", "replicate"]).any()
    summary = raw.groupby(["n", "p", "nominal_gamma", "gamma", "a", "k"], as_index=False).agg(
        mean_error=("error", "mean"), sd_error=("error", "std"), repeats=("error", "size"))
    summary["mcse"] = summary.sd_error / np.sqrt(summary.repeats)
    raw.to_csv(DATA / "raw.csv", index=False)
    summary.to_csv(DATA / "summary.csv", index=False)
    report = dict(passed=True, n=n, repeats=repeats, seed=seed, total_signal=TOTAL,
                  sigma2=SIGMA2, exponents=list(EXPONENTS), components=list(COUNTS), ols_code=0,
                  grid="36 log-equally-spaced targets in [0.1,10]; p=round(n*gamma); actual p/n plotted",
                  p_values=dimensions.tolist(), nominal_gamma=nominal.tolist(),
                  model="X_ij,w_i,epsilon_i iid N(0,1); beta=sqrt(5-b)*e_1; y=X*beta+sqrt(b)*w+epsilon; b=5*(1+p/n)^(-a)",
                  pairing="Same X, omitted predictor w, and noise for both a values and all estimators",
                  criterion="squared coefficient error plus b, without test-response noise",
                  independent_training_samples=36 * repeats, raw_rows=len(raw),
                  max_independent_fit_relative_difference=fit_error,
                  elapsed_seconds=time.perf_counter() - started,
                  files_sha256={name: digest(DATA / name) for name in ("raw.csv", "summary.csv")},
                  source_sha256={name: digest(Path(__file__).with_name(name)) for name in SOURCES})
    save_json(DATA / "simulation_manifest.json", report)


def validate_theory():
    maximum, refinement, single, ols_error = 0.0, 0.0, 0.0, 0.0
    for g in (.1, .25, .5, .9, 1., 1.1, 2., 2.5, 10.):
        for a in EXPONENTS:
            signal, noise, missing = parameters(g, a)
            path = limiting_path(g, a)
            refinement = max(refinement, float(np.max(abs(path - limiting_path(g, a, 128)))))
            single = max(single, abs(path[0] - missing - float(pls1_candidate_risk(g, signal, noise))))
            for k in COUNTS:
                reference = missing + noise * high_precision_risk(g, signal / noise, k)
                maximum = max(maximum, abs(path[k - 1] - reference))
            if g != 1:
                explicit = missing + (noise * g / (1-g) if g < 1 else signal*(1-1/g)+noise/(g-1))
                ols_error = max(ols_error, abs(ols_limit(g, a) - explicit))
    assert maximum < 1e-8 and refinement < 1e-8 and single < 1e-10 and ols_error < 1e-10
    return dict(max_80_digit_difference=maximum, max_64_vs_128_node_difference=refinement,
                max_pls1_closed_form_difference=single, max_ols_formula_difference=ols_error,
                high_precision_checks=9 * len(EXPONENTS) * len(COUNTS))


def plot_and_audit():
    simulation = json.loads((DATA / "simulation_manifest.json").read_text(encoding="utf-8"))
    assert simulation["passed"] and simulation["exponents"] == list(EXPONENTS)
    for name, value in simulation["files_sha256"].items():
        assert digest(DATA / name) == value
    for name, value in simulation["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == value
    raw = pd.read_csv(DATA / "raw.csv")
    summary = pd.read_csv(DATA / "summary.csv")
    recalculated = raw.groupby(["p", "a", "k"], as_index=False).error.mean()
    checked = summary.merge(recalculated, on=["p", "a", "k"], validate="one_to_one")
    np.testing.assert_allclose(checked.mean_error, checked.error, rtol=1e-12, atol=1e-12)
    assert summary.groupby(["a", "k"]).size().eq(36).all()
    assert summary.repeats.eq(simulation["repeats"]).all()
    checks = validate_theory()
    gamma = np.unique(np.r_[np.geomspace(.1, 10, 1201), np.linspace(.8, 1.2, 201), 1.0])
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
                         "axes.labelsize": 13, "legend.fontsize": 10})
    artifacts, records, shapes, diagnostics, figure_names = [], [], [], [], []
    for a in EXPONENTS:
        curves = np.array([limiting_path(g, a) for g in gamma])
        assert np.isfinite(curves).all() and curves.min() >= 0
        fig, ax = plt.subplots(figsize=(7.2, 5.6), layout="constrained")
        for k, color in zip(COUNTS, ("black", "red", "#00B800", "blue")):
            y = curves[:, k - 1]
            ax.plot(gamma, y, color=color, lw=1.65, marker="o", markevery=[], ms=3.5,
                    label=rf"PLS $k={k}$")
            points = summary[(summary.a == a) & (summary.k == k)].sort_values("gamma")
            ax.plot(points.gamma, points.mean_error, ls="none", marker="o", ms=3.5, color=color, zorder=3)
            records.extend((float(g), a, k, float(value)) for g, value in zip(gamma, y))
            extrema = []
            for kind, sign in (("minimum", 1), ("maximum", -1)):
                for index in find_peaks(-sign * y, prominence=1e-6)[0]:
                    result = minimize_scalar(lambda g: sign * limiting_path(g, a)[k - 1],
                                             bounds=(gamma[index-1], gamma[index+1]), method="bounded")
                    extrema.append(dict(kind=kind, gamma=float(result.x), error=float(sign * result.fun)))
            shapes.append(dict(a=a, k=k, extrema=sorted(extrema, key=lambda item: item["gamma"])))
            theoretical = np.array([limiting_path(g, a)[k-1] for g in points.gamma])
            for row, value in zip(points.itertuples(), theoretical):
                diagnostics.append(dict(a=a, k=k, gamma=row.gamma, theory=value,
                                        mean=row.mean_error, mcse=row.mcse,
                                        relative_difference=(row.mean_error-value)/value))
        for mask, label in ((gamma < 1, "OLS"), (gamma > 1, None)):
            branch = np.array([ols_limit(g, a) for g in gamma[mask]])
            ax.plot(gamma[mask], branch, color="0.35", ls="--", lw=1.6,
                    marker="o", markevery=[], ms=3.5, markerfacecolor="white", label=label)
            records.extend((float(g), a, 0, float(value)) for g, value in zip(gamma[mask], branch))
        points = summary[(summary.a == a) & (summary.k == 0)].sort_values("gamma")
        ax.plot(points.gamma, points.mean_error, ls="none", marker="o", ms=3.5,
                markerfacecolor="white", color="0.35", zorder=4)
        ax.axhline(TOTAL, color="0.5", ls=":", lw=1.2, label="Null")
        ax.axvline(1, color="0.7", ls=":", lw=.8, zorder=0)
        upper = 9 if a == 2 else 7
        ax.set(xscale="log", xlim=(.085, 12), ylim=(0, upper),
               xlabel=r"$\gamma$", ylabel="Generalization error")
        assert not ax.get_title()
        ax.xaxis.set_major_locator(FixedLocator([.1, .2, .5, 1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(loc="upper left", frameon=True, fancybox=False,
                  edgecolor="black", facecolor="white", framealpha=1)
        assert np.max(curves[:, np.array(COUNTS)-1]) < upper
        assert summary[(summary.a == a) & (summary.k != 0)].mean_error.between(0, upper).all()
        name = f"misspecified_pls_ols_a_{a}"
        figure_names.append(name)
        path = FIGURES / f"{name}.png"
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
        destination = ROOT / "Paper/Figs" / path.name
        shutil.copyfile(path, destination)
        assert digest(path) == digest(destination)
        artifacts.append(path)
    table = DATA / "theory.csv"
    pd.DataFrame(records, columns=["gamma", "a", "k", "error"]).to_csv(table, index=False)
    artifacts.append(table)
    pd.DataFrame(diagnostics).to_csv(DATA / "monte_carlo_comparison.csv", index=False)
    artifacts.append(DATA / "monte_carlo_comparison.csv")
    signatures = {(row["a"], row["k"]): [entry["kind"] for entry in row["extrema"]] for row in shapes}
    assert signatures[(2, 1)] == signatures[(2, 2)] == ["minimum"]
    assert signatures[(2, 5)] == signatures[(2, 10)] == ["minimum", "maximum", "minimum"]
    assert signatures[(5, 1)] == signatures[(5, 2)] == signatures[(5, 5)] == ["minimum"]
    assert signatures[(5, 10)] == ["minimum", "maximum", "minimum"]
    report = dict(passed=True, figure_names=figure_names, total_signal=TOTAL, sigma2=SIGMA2,
                  exponents=list(EXPONENTS), components=list(COUNTS), title="",
                  xscale="log", yscale="linear", ylim_by_exponent={"2": [0, 9], "5": [0, 7]},
                  ols_at_gamma_one="divergent; branches plotted separately; no finite value assigned",
                  checks=checks, numerical_extrema_on_displayed_range=shapes,
                  max_pls_relative_mean_difference=max(abs(row["relative_difference"]) for row in diagnostics),
                  simulation_manifest_sha256=digest(DATA / "simulation_manifest.json"),
                  files_sha256={path.relative_to(ROOT).as_posix(): digest(path) for path in artifacts},
                  source_sha256={name: digest(Path(__file__).with_name(name)) for name in SOURCES})
    save_json(DATA / "figure_manifest.json", report)
    print(json.dumps({"checks": checks, "numerical_shapes": shapes,
                      "max_pls_relative_mean_difference": report["max_pls_relative_mean_difference"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026092802)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        simulate(args.n, args.repeats, args.seed)
    plot_and_audit()


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
