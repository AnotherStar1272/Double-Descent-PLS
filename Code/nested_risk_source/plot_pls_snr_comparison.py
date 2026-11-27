"""Four figures: theoretical limits and Monte Carlo means for PLS and OLS.

Uses exact polynomial quadrature for lambda*dMP and a diagonal CG recurrence,
with independent 80-decimal moment-system checks. Read audited simulations
from outputs_snr_comparison; write only PNGs to theory/.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
from hashlib import sha256
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import mpmath as mp
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from models import _cg_diagonal_filters, ols_asymptotic_risk, pls1_candidate_risk

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Experiment/theory"
DATA = ROOT / "Experiment/outputs_snr_comparison"
SNRS = (1.0, 2.33, 3.66, 5.0)
COUNTS = (1, 2, 5, 10)
NAMES = ("pls_ols_snr_1", "pls_ols_snr_2_33", "pls_ols_snr_3_66", "pls_ols_snr_5")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def risk_path(gamma, snr, nodes=64):
    """Integrate polynomial risks against lambda*dMP, including the zero atom.

    Under lambda=1+gamma+2*sqrt(gamma)*cos(theta), lambda*dMP becomes
    2/pi*sin(theta)^2*dtheta, a probability measure. N-point Chebyshev-U
    quadrature is exact through degree 2N-1. Expanding the signal term
    as 1-2*int(lambda*q)+int(lambda^2*q^2) retains the zero atom automatically.
    """
    angles = np.arange(1, nodes + 1) * np.pi / (nodes + 1)
    lam = 1 + gamma + 2 * np.sqrt(gamma) * np.cos(angles)
    weights = 2 / (nodes + 1) * np.sin(angles)**2
    rhs = np.sqrt(weights * (snr * lam + gamma))
    filters = _cg_diagonal_filters(lam, rhs, max(COUNTS))
    return np.array([snr * (1 - 2 * (weights @ q) + weights @ (lam * q*q))
                     + gamma * (weights @ (q*q)) for q in filters])


def high_precision_risk(gamma, snr, k):
    """Independent 80-digit solution of the manuscript's moment equations."""
    with mp.workdps(80):
        g, r = mp.mpf(str(gamma)), mp.mpf(str(snr))
        moments = [mp.mpf(1)] + [sum(mp.mpf(comb(j, ell)*comb(j, ell-1))
                    * g**(ell-1)/j for ell in range(1, j+1))
                    for j in range(1, 2*k+2)]
        A = mp.matrix(k, k)
        B = mp.matrix(k, k)
        a = mp.matrix(k, 1)
        b = mp.matrix(k, 1)
        for i in range(k):
            a[i] = r*moments[i+2] + g*moments[i+1]
            b[i] = r*moments[i+1]
            for j in range(k):
                A[i, j] = r*moments[i+j+3] + g*moments[i+j+2]
                B[i, j] = r*moments[i+j+2] + g*moments[i+j+1]
        eta = mp.lu_solve(A, a)
        return float(r - 2*(eta.T*b)[0] + (eta.T*B*eta)[0])


def validate():
    error = 0.0
    refinement = 0.0
    single = 0.0
    for g in (0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 10.0):
        for snr in SNRS:
            path = risk_path(g, snr)
            refinement = max(refinement, float(np.max(abs(path-risk_path(g, snr, 128)))))
            single = max(single, abs(path[0] - float(pls1_candidate_risk(g, snr, 1))))
            for k in COUNTS:
                error = max(error, abs(path[k-1] - high_precision_risk(g, snr, k)))
    assert error < 1e-9 and refinement < 1e-9 and single < 1e-10
    return dict(max_80_digit_absolute_difference=error,
                max_64_vs_128_node_difference=refinement,
                max_pls1_closed_form_difference=single,
                high_precision_checks=9*len(SNRS)*len(COUNTS))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    simulation_path = DATA/"simulation_manifest.json"
    simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
    assert simulation["passed"] and simulation["snrs"] == list(SNRS)
    for name, value in simulation["files_sha256"].items():
        assert digest(DATA/name) == value, f"Changed simulation data: {name}"
    for name, value in simulation["source_sha256"].items():
        assert digest(Path(__file__).parent/name) == value, f"Changed simulation source: {name}"
    summary = pd.read_csv(DATA/"summary.csv")
    assert summary.groupby(["snr", "k"]).size().eq(36).all()
    assert summary.repeats.eq(simulation["repeats"]).all()
    checks = validate()
    gamma = np.unique(np.r_[np.geomspace(.1, 10, 901), np.linspace(.8, 1.2, 201), 1.0])
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 11, "axes.labelsize": 13, "legend.fontsize": 10,
                         "pdf.fonttype": 42})
    artifacts = []
    records = []
    clipped = {}
    for snr, name in zip(SNRS, NAMES):
        curves = np.array([risk_path(g, snr) for g in gamma])
        assert np.isfinite(curves).all() and np.min(curves) >= 0
        ols = ols_asymptotic_risk(gamma, snr, 1)
        fig, ax = plt.subplots(figsize=(7.2, 5.6), layout="constrained")
        for k, color in zip(COUNTS, ("black", "red", "#00B800", "blue")):
            ax.plot(gamma, curves[:, k-1], color=color, lw=1.65,
                    marker="o", markevery=[], ms=3.5, label=rf"PLS $k={k}$")
            points = summary[(summary.snr == snr) & (summary.k == k)].sort_values("gamma")
            ax.plot(points.gamma, points.mean_risk, ls="none", marker="o", ms=3.5,
                    color=color, zorder=3)
            records.extend((float(g), snr, k, float(r)) for g, r in zip(gamma, curves[:, k-1]))
        for mask, label in ((gamma < 1, "OLS"), (gamma > 1, None)):
            ax.plot(gamma[mask], ols[mask], color="0.35", ls="--", lw=1.6,
                    marker="o", markevery=[], ms=3.5, markerfacecolor="white", label=label)
        ols_points = summary[(summary.snr == snr) & (summary.k == 0)].sort_values("gamma")
        ax.plot(ols_points.gamma, ols_points.mean_risk, ls="none", marker="o", ms=3.5,
                markerfacecolor="white", color="0.35", zorder=4)
        # No finite OLS value or line segment is fabricated at gamma=1.
        records.extend((float(g), snr, 0, float(r)) for g, r in zip(gamma, ols) if g != 1)
        ax.axhline(snr, color="0.5", ls=":", lw=1.2, label="Null")
        ax.axvline(1, color="0.7", ls=":", lw=.8, zorder=0)
        upper = 7 if snr == 5 else 5
        ax.set(xscale="log", xlim=(.085, 12), ylim=(0, upper),
               xlabel=r"$\gamma$", ylabel="Risk")
        assert not ax.get_title()
        ax.xaxis.set_major_locator(FixedLocator([.1, .2, .5, 1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(loc="upper left", frameon=True, fancybox=False,
                  edgecolor="black", facecolor="white", framealpha=1)
        assert np.max(curves[:, np.array(COUNTS)-1]) < upper
        shown = summary[summary.snr == snr]
        assert shown[shown.k != 0].mean_risk.between(0, upper).all()
        clipped[str(snr)] = int((ols_points.mean_risk > upper).sum())
        path = OUT/f"{name}.png"
        fig.savefig(path, dpi=300, facecolor="white")
        artifacts.append(path)
        plt.close(fig)
    table = DATA/"pls_ols_snr_theory.csv"
    np.savetxt(table, np.array(records), delimiter=",", header="gamma,snr,k,risk", comments="",
               fmt=["%.17g", "%.17g", "%d", "%.17g"])
    artifacts.append(table)
    report = dict(passed=True, figure_names=list(NAMES), snrs=list(SNRS), components=list(COUNTS),
                  sigma2=1, simulation_manifest_sha256=digest(simulation_path),
                  title="", xscale="log", yscale="linear",
                  ylim_by_snr={str(snr):[0, 7 if snr == 5 else 5] for snr in SNRS},
                  clipped_ols_mean_points=clipped,
                  ols_at_gamma_one="diverges; omitted from CSV and split into two branches",
                  ols_source="Hastie2022; manuscript eq:ols-isotropic",
                  quadrature="64-node Gauss-Chebyshev U for lambda*dMP; diagonal CG recurrence",
                  checks=checks, files_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in artifacts},
                  source_sha256={p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("models.py"))})
    (DATA/"pls_ols_snr_manifest.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print("Created four title-free PLS/OLS figures in", OUT)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
