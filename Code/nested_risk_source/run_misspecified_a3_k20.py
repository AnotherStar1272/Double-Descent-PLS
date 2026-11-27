"""Selected omitted-feature comparison: a=3, k=1,2,5,10,20 and OLS.

The paired seeds and grid match the earlier a=3,4 candidates. A new data
directory and figure basename preserve those candidate calculations.
Run once; use --plot-only to redraw and re-audit saved simulations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_misspecified_snr_comparison as comparison
import plot_pls_snr_comparison as theory
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
from scipy.linalg import eigh, lstsq
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Experiment/outputs_misspecified_a3_k20"
FIGURES = ROOT / "Experiment/theory"
COUNTS = (1, 2, 5, 10, 20)
SOURCES = (Path(__file__).name, *comparison.SOURCES)
NAME = "misspecified_pls_ols_a_3_k20"


def spectral_estimates(lam, rhs):
    """PLS least squares in a two-pass orthogonalized spectral Krylov basis."""
    columns = []
    vector = rhs.copy()
    threshold = 1e-13 * max(1., float(np.linalg.norm(rhs)))
    for _ in range(min(max(COUNTS), len(lam))):
        if columns:
            Q = np.column_stack(columns)
            for _pass in range(2):
                vector -= Q @ (Q.T @ vector)
        size = np.linalg.norm(vector)
        if size <= threshold:
            break
        columns.append(vector / size)
        vector = lam * columns[-1]
    Q = np.column_stack(columns)
    fitted = {}
    for k in COUNTS:
        basis = Q[:, :min(k, Q.shape[1])]
        coordinates = lstsq(np.sqrt(lam)[:, None] * basis, rhs / np.sqrt(lam),
                            lapack_driver="gelsy", check_finite=False)[0]
        fitted[k] = basis @ coordinates
    return fitted


def sample_errors(X, omitted, noise):
    n, p = X.shape
    signal, _, missing = comparison.parameters(p / n, 3)
    errors = np.sqrt(missing) * omitted + noise
    if p <= n:
        lam, U = eigh(X.T @ X / n, check_finite=False)
        beta = U[0] * np.sqrt(signal)
        rhs = lam * beta + U.T @ (X.T @ errors / n)
        null_error = 0.
    else:
        lam, U = eigh(X @ X.T / n, check_finite=False)
        projected = U.T @ X[:, 0]
        beta = projected / np.sqrt(n * lam) * np.sqrt(signal)
        rhs = np.sqrt(lam / n) * (projected * np.sqrt(signal) + U.T @ errors)
        null_error = signal - float(beta @ beta)
        assert null_error > -1e-9
        null_error = max(0., null_error)
    assert lam.min() > 0
    fitted = spectral_estimates(lam, rhs)
    result = [float(np.sum((fitted[k] - beta)**2)) + null_error + missing for k in COUNTS]
    result.append(float(np.sum((rhs / lam - beta)**2)) + null_error + missing)
    return np.array([result])


def configure():
    comparison.DATA = DATA
    comparison.EXPONENTS = (3,)
    comparison.COUNTS = COUNTS
    comparison.SOURCES = SOURCES
    comparison.sample_errors = sample_errors
    theory.COUNTS = COUNTS


def endpoint_check():
    """Check coefficient equality with a direct OLS solve at n=200,p=k=20."""
    maximum = 0.
    for rep in range(5):
        rng = np.random.default_rng(np.random.SeedSequence([2026092802, 200, 20, rep]))
        X = rng.normal(size=(200, 20))
        omitted, noise = rng.normal(size=(2, 200))
        signal, _, missing = comparison.parameters(.1, 3)
        y = np.sqrt(signal) * X[:, 0] + np.sqrt(missing) * omitted + noise
        lam, U = eigh(X.T @ X / 200)
        rhs = U.T @ (X.T @ y / 200)
        estimate = U @ spectral_estimates(lam, rhs)[20]
        reference = comparison.minnorm_ols(X, y)
        maximum = max(maximum, float(np.linalg.norm(estimate-reference)))
    assert maximum < 1e-10
    return maximum


def plot_and_audit():
    simulation = json.loads((DATA / "simulation_manifest.json").read_text(encoding="utf-8"))
    assert simulation["passed"] and simulation["exponents"] == [3]
    assert simulation["components"] == list(COUNTS)
    for name, value in simulation["files_sha256"].items():
        assert comparison.digest(DATA / name) == value
    for name, value in simulation["source_sha256"].items():
        assert comparison.digest(Path(__file__).with_name(name)) == value
    raw = pd.read_csv(DATA / "raw.csv")
    summary = pd.read_csv(DATA / "summary.csv")
    means = raw.groupby(["p", "a", "k"], as_index=False).error.mean()
    checked = summary.merge(means, on=["p", "a", "k"], validate="one_to_one")
    np.testing.assert_allclose(checked.mean_error, checked.error, rtol=1e-12, atol=1e-12)
    assert summary.groupby("k").size().eq(36).all()
    assert summary.repeats.eq(simulation["repeats"]).all()
    # An independent run with the same seed must reproduce all earlier points.
    previous_path = ROOT / "Experiment/outputs_misspecified_candidates/raw.csv"
    previous_manifest = json.loads(previous_path.with_name("simulation_manifest.json").read_text(encoding="utf-8"))
    paired = None
    if all(simulation[key] == previous_manifest[key] for key in ("n", "repeats", "seed")):
        assert comparison.digest(previous_path) == previous_manifest["files_sha256"]["raw.csv"]
        previous = pd.read_csv(previous_path)
        paired_rows = raw.merge(previous[previous.a == 3], on=["n", "p", "a", "k", "replicate"],
                                suffixes=("_new", "_previous"), validate="one_to_one")
        assert len(paired_rows) == 36 * simulation["repeats"] * 5
        paired = float(np.max(abs(paired_rows.error_new-paired_rows.error_previous)))
        assert paired < 1e-9
    endpoint = raw[raw.p == 20].pivot(index="replicate", columns="k", values="error")
    endpoint_difference = float(np.max(abs(endpoint[20]-endpoint[0]))) if len(endpoint) else None
    if endpoint_difference is not None:
        assert endpoint_difference < 1e-10
    checks = comparison.validate_theory()
    checks.update(max_p20_coefficient_difference_from_ols=endpoint_check(),
                  max_p20_sample_error_difference_from_ols=endpoint_difference,
                  max_paired_difference_from_previous_run=paired)
    gamma = np.unique(np.r_[np.geomspace(.1, 10, 1201), np.linspace(.8, 1.2, 201), 1.])
    curves = np.array([comparison.limiting_path(g, 3) for g in gamma])
    assert np.isfinite(curves).all() and curves.min() >= 0
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
                         "axes.labelsize": 13, "legend.fontsize": 10})
    fig, ax = plt.subplots(figsize=(7.2, 5.6), layout="constrained")
    records, shapes, diagnostics = [], [], []
    for k, color in zip(COUNTS, ("black", "red", "#00B800", "blue", "#A020B0")):
        values = curves[:, k-1]
        ax.plot(gamma, values, color=color, lw=1.65, marker="o", markevery=[], ms=3.5,
                label=rf"PLS $k={k}$")
        points = summary[summary.k == k].sort_values("gamma")
        ax.plot(points.gamma, points.mean_error, ls="none", marker="o", ms=3.5, color=color, zorder=3)
        records.extend((float(g), 3, k, float(value)) for g, value in zip(gamma, values))
        extrema = []
        for kind, sign in (("minimum", 1), ("maximum", -1)):
            for index in find_peaks(-sign * values, prominence=1e-6)[0]:
                result = minimize_scalar(lambda g: sign * comparison.limiting_path(g, 3)[k-1],
                                         bounds=(gamma[index-1], gamma[index+1]), method="bounded")
                extrema.append(dict(kind=kind, gamma=float(result.x), error=float(sign * result.fun)))
        shapes.append(dict(a=3, k=k, extrema=sorted(extrema, key=lambda row: row["gamma"])))
        for row in points.itertuples():
            value = float(comparison.limiting_path(row.gamma, 3)[k-1])
            diagnostics.append(dict(a=3, k=k, gamma=row.gamma, theory=value,
                                    mean=row.mean_error, mcse=row.mcse,
                                    relative_difference=(row.mean_error-value)/value))
    for mask, label in ((gamma < 1, "OLS"), (gamma > 1, None)):
        branch = np.array([comparison.ols_limit(g, 3) for g in gamma[mask]])
        ax.plot(gamma[mask], branch, color="0.35", ls="--", lw=1.6,
                marker="o", markevery=[], ms=3.5, markerfacecolor="white", label=label)
        records.extend((float(g), 3, 0, float(value)) for g, value in zip(gamma[mask], branch))
    points = summary[summary.k == 0].sort_values("gamma")
    ax.plot(points.gamma, points.mean_error, ls="none", marker="o", ms=3.5,
            markerfacecolor="white", color="0.35", zorder=4)
    ax.axhline(5, color="0.5", ls=":", lw=1.2, label="Null")
    ax.axvline(1, color="0.7", ls=":", lw=.8, zorder=0)
    ax.set(xscale="log", xlim=(.085, 12), ylim=(0, 12), xlabel=r"$\gamma$", ylabel="Generalization error")
    ax.xaxis.set_major_locator(FixedLocator([.1, .2, .5, 1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black", facecolor="white", framealpha=1)
    assert not ax.get_title()
    assert np.max(curves[:, np.array(COUNTS)-1]) < 12
    assert summary[summary.k != 0].mean_error.between(0, 12).all()
    path = FIGURES / f"{NAME}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    table = DATA / "theory.csv"
    pd.DataFrame(records, columns=["gamma", "a", "k", "error"]).to_csv(table, index=False)
    diagnostics_path = DATA / "monte_carlo_comparison.csv"
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    report = dict(passed=True, figure_names=[NAME], total_signal=5., sigma2=1.,
                  exponents=[3], components=list(COUNTS), title="", xscale="log", yscale="linear",
                  ylim_by_exponent={"3": [0, 12]}, checks=checks,
                  numerical_extrema_on_displayed_range=shapes,
                  max_pls_relative_mean_difference=max(abs(row["relative_difference"]) for row in diagnostics),
                  sample_algorithm="two-pass orthogonalized spectral Krylov basis and QR least squares",
                  simulation_manifest_sha256=comparison.digest(DATA / "simulation_manifest.json"),
                  files_sha256={item.relative_to(ROOT).as_posix(): comparison.digest(item)
                                for item in (path, table, diagnostics_path)},
                  source_sha256={name: comparison.digest(Path(__file__).with_name(name)) for name in SOURCES})
    comparison.save_json(DATA / "figure_manifest.json", report)
    print(json.dumps({"checks": checks, "numerical_shapes": shapes,
                      "max_pls_relative_mean_difference": report["max_pls_relative_mean_difference"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026092802)
    args = parser.parse_args()
    configure()
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        comparison.simulate(args.n, args.repeats, args.seed)
        manifest_path = DATA / "simulation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["pairing"] = "Same X, omitted predictor w, and noise for all estimators at a=3; seeds match the earlier candidate runs"
        manifest["sample_algorithm"] = "two-pass orthogonalized spectral Krylov basis and QR least squares"
        comparison.save_json(manifest_path, manifest)
    plot_and_audit()


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
