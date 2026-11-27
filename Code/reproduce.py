"""Generate the manuscript's numerical results and vector figures."""
import argparse
import csv
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#111111", "#505050", "#707070")
BENCHMARK_SNRS = (1, 5, 10, 20, 40)
LINE_STYLES = ((), (5, 3), (1, 3))
MARKERS = ("circle", "square", "diamond")
VALIDATION_STYLES = (*zip(COLORS, LINE_STYLES, MARKERS), ("#333333", (5, 2, 1, 2), "triangle"))

def figure_canvas(path, pagesize):
    """Embed the regular and bold Bitstream Vera fonts bundled with ReportLab."""
    pdfmetrics.registerFont(TTFont("Vera", "Vera.ttf"))
    pdfmetrics.registerFont(TTFont("Vera-Bold", "VeraBd.ttf"))
    return canvas.Canvas(str(path), pagesize=pagesize, invariant=1, initialFontName="Vera")


def isotropic(gamma, snr, count=40, nodes=512):
    """Integrate the limiting PLS path with MP quadrature and diagonal CG."""
    theta = np.pi * np.arange(1, nodes + 1) / (nodes + 1)
    spectrum = (1 - np.sqrt(gamma)) ** 2 + 4 * np.sqrt(gamma) * np.sin(theta / 2) ** 2
    weights = 2 * np.sin(theta) ** 2 / (nodes + 1) / spectrum
    atom = max(1 - 1 / gamma, 0)
    rhs = np.sqrt(weights * spectrum * (snr * spectrum + gamma))
    coefficients = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual.copy()
    norm = residual @ residual
    risks = []
    for _ in range(count):
        if norm > 1e-28:
            product = spectrum * direction
            step = norm / (direction @ product)
            coefficients += step * direction
            residual -= step * product
            updated = residual @ residual
            direction = residual + updated / norm * direction
            norm = updated
        polynomial = coefficients / rhs
        risks.append(
            snr * (atom + weights @ (1 - spectrum * polynomial) ** 2)
            + gamma * (weights * spectrum) @ polynomial**2
        )
    ridge = snr * atom + weights @ (gamma * snr / (gamma + snr * spectrum))
    if gamma == 1:
        # The quadrature exactly integrates t dF(t), but loses O(1/nodes)
        # mass for dF(t) at the zero endpoint. Both risks share this offset.
        correction = snr * (1 - weights.sum())
        risks = np.asarray(risks) + correction
        ridge += correction
    ols = (
        gamma / (1 - gamma) if gamma < 1
        else snr * atom + 1 / (gamma - 1) if gamma > 1
        else np.inf
    )
    return np.asarray(risks), ridge, ols


def pls(x, y, count):
    """Return iterates of conjugate gradients for the PLS normal equations."""
    residual = x.T @ y / len(y)
    direction = residual.copy()
    beta = np.zeros_like(residual)
    norm = residual @ residual
    for _ in range(count):
        if norm > 1e-24:
            projected = x @ direction
            step = norm / (projected @ projected / len(y))
            beta += step * direction
            residual -= step * (x.T @ projected / len(y))
            updated = residual @ residual
            direction = residual + updated / norm * direction
            norm = updated
        yield beta.copy()


def summary(values):
    values = np.asarray(values)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def benchmark(rng, reps, n, snrs=(1, 10, 40)):
    data = []
    for gamma in (0.5, 1.0, 2.0):
        p = round(gamma * n)
        for snr in snrs:
            path, limit_ridge, limit_ols = isotropic(gamma, snr, nodes=1024)
            count = int(path.argmin()) + 1
            beta = np.zeros(p)
            beta[0] = np.sqrt(snr)
            errors = []
            for _ in range(reps):
                x = rng.standard_normal((n, p))
                y = beta[0] * x[:, 0] + rng.standard_normal(n)
                fit = list(pls(x, y, count))[-1]
                if p <= n:
                    gram = x.T @ x
                    rhs = x.T @ y
                    least_squares = np.linalg.solve(gram, rhs) if gamma != 1 else None
                    gram.flat[::p + 1] += n * gamma / snr
                    ridge = np.linalg.solve(gram, rhs)
                else:
                    gram = x @ x.T
                    least_squares = x.T @ np.linalg.solve(gram, y)
                    gram.flat[::n + 1] += n * gamma / snr
                    ridge = x.T @ np.linalg.solve(gram, y)
                errors.append([np.sum((fit - beta)**2), np.sum((ridge - beta)**2),
                               np.sum((least_squares - beta)**2) if gamma != 1 else np.nan])
            errors = np.asarray(errors)
            means, se = summary(errors)
            paired_means, paired_se = summary(errors[:, :1] - errors[:, 1:])
            data.append((gamma, snr, count, path[count-1], limit_ridge, limit_ols,
                         *means, *se, *paired_means, *paired_se))
    return data


def loss_gap(x_train, y_train, x_test, y_test):
    one, two = pls(x_train, y_train, 2)
    residual = y_test - x_test @ one
    change = x_test @ (two - one)
    return np.mean(change * (change - 2 * residual))


def loo_two_count_predictions(x, y):
    """Return exact leave-one-out predictions for one and two PLS components."""
    n = len(y)
    # A shared divisor n leaves each training Krylov space unchanged.
    gram = x @ x.T / n
    gram_squared = gram @ gram
    diagonal = (
        gram.diagonal(),
        np.einsum("ij,ij->i", gram, gram),
        np.einsum("ij,ij->i", gram, gram_squared),
        np.einsum("ij,ij->i", gram_squared, gram_squared),
    )
    responses = [y]
    for _ in range(4):
        responses.append(gram @ responses[-1])

    # The coefficient of z^r in y' (I-zK)^(-1) y gives y' K^r y.
    # Deleting row i subtracts the square of [(I-zK)^(-1)y]_i divided
    # by [(I-zK)^(-1)]_{ii}. Divide those series up to degree four.
    corrections = [y**2]
    moments = []
    for order in range(1, 5):
        cross_products = sum(responses[a] * responses[order - a] for a in range(order + 1))
        lower_orders = sum(diagonal[a - 1] * corrections[order - a] for a in range(1, order + 1))
        corrections.append(cross_products - lower_orders)
        moments.append((y @ responses[order] - corrections[-1]) / n)

    s0, s1, s2, s3 = moments
    first = responses[1] - diagonal[0] * y
    second = responses[2] - diagonal[1] * y - diagonal[0] * first
    determinant = s1 * s3 - s2**2
    if np.any(s1 <= 0) or np.any(determinant <= 0):
        raise ValueError("The leave-one-out two-component normal equations must be nonsingular.")
    one = first * s0 / s1
    two = (first * (s0 * s3 - s1 * s2) + second * (s1**2 - s0 * s2)) / determinant
    return np.column_stack((one, two))


def validation(rng, reps, sizes):
    data = []
    for n in sizes:
        p = round(0.9 * n)
        beta = np.zeros(p)
        beta[0] = np.sqrt(1.5)
        v = int(n ** (2 / 3))
        gaps = []
        for _ in range(reps):
            x = rng.standard_normal((n, p))
            y = beta[0] * x[:, 0] + rng.standard_normal(n)
            one, two = pls(x, y, 2)
            full = np.sum((two - beta)**2) - np.sum((one - beta)**2)
            holdout = loss_gap(x[:-v], y[:-v], x[-v:], y[-v:])
            folds = []
            for test in np.array_split(np.arange(n), 5):
                train = np.ones(n, dtype=bool)
                train[test] = False
                folds.append(loss_gap(x[train], y[train], x[test], y[test]))
            leave_one_out = loo_two_count_predictions(x, y)
            change = leave_one_out[:, 1] - leave_one_out[:, 0]
            loo = np.mean(change * (change - 2 * (y - leave_one_out[:, 0])))
            gaps.append((full, holdout, np.mean(folds), loo))
        means, se = summary(gaps)
        fraction = np.mean(np.asarray(gaps) < 0, axis=0)
        data.append((n, v, *means[:3], *se[:3], *fraction[:3], means[3], se[3], fraction[3]))
    return data


def write_csv(folder, name, headers, rows):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / name).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def axes(pdf, left, bottom, width, height, x_ticks, y_ticks, x_label, y_label,
         x_tick_labels=None, x_limits=None, y_tick_labels=None):
    x_lower, x_upper = x_limits if x_limits is not None else (x_ticks[0], x_ticks[-1])

    def point(x, y):
        return (left + width * (x - x_lower) / (x_upper - x_lower),
                bottom + height * (y - y_ticks[0]) / (y_ticks[-1] - y_ticks[0]))

    pdf.setFont("Vera", 10)
    pdf.setStrokeColor(HexColor("#e0e0e0"))
    pdf.setLineWidth(0.45)
    for y, label in zip(y_ticks, y_tick_labels or [f"{y:g}" for y in y_ticks]):
        x0, y0 = point(x_lower, y)
        pdf.line(x0, y0, x0 + width, y0)
        pdf.drawRightString(x0 - 5, y0 - 3, str(label))
    pdf.setStrokeColor(HexColor("#303030"))
    pdf.line(left, bottom, left + width, bottom)
    pdf.line(left, bottom, left, bottom + height)
    for x, label in zip(x_ticks, x_tick_labels or x_ticks):
        x0, y0 = point(x, y_ticks[0])
        pdf.line(x0, y0, x0, y0 - 3)
        pdf.drawCentredString(x0, y0 - 15, str(label))
    pdf.drawCentredString(left + width / 2, bottom - 30, x_label)
    pdf.saveState()
    pdf.translate(left - 40, bottom + height / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, y_label)
    pdf.restoreState()
    return point


def series(pdf, point, pairs, color, dash=(), errors=None, marker=None):
    pdf.setStrokeColor(HexColor(color))
    pdf.setFillColor(HexColor(color))
    pdf.setLineWidth(1.3)
    pdf.setDash(dash)
    coords = [point(*pair) for pair in pairs]
    if coords:
        path = pdf.beginPath()
        path.moveTo(*coords[0])
        for position in coords[1:]:
            path.lineTo(*position)
        pdf.drawPath(path)
    pdf.setDash()
    for x, y in coords:
        if marker == "circle":
            pdf.circle(x, y, 2.3, fill=1, stroke=0)
        elif marker == "square":
            pdf.rect(x - 2.1, y - 2.1, 4.2, 4.2, fill=1, stroke=0)
        elif marker == "diamond":
            path = pdf.beginPath()
            path.moveTo(x - 2.8, y)
            path.lineTo(x, y + 2.8)
            path.lineTo(x + 2.8, y)
            path.lineTo(x, y - 2.8)
            path.close()
            pdf.drawPath(path, fill=1, stroke=0)
        elif marker == "triangle":
            path = pdf.beginPath()
            path.moveTo(x, y + 3)
            path.lineTo(x - 2.9, y - 2.3)
            path.lineTo(x + 2.9, y - 2.3)
            path.close()
            pdf.drawPath(path, fill=1, stroke=0)

    if errors is not None:
        for (x, y), e in zip(pairs, errors):
            px, lo = point(x, y - 2 * e)
            _, hi = point(x, y + 2 * e)
            pdf.line(px, lo, px, hi)
            pdf.line(px - 3, lo, px + 3, lo)
            pdf.line(px - 3, hi, px + 3, hi)
    pdf.setFillColor(HexColor("#181818"))



def figures(folder, benchmark_rows, validation_rows):
    pdf = figure_canvas(folder / "risk.pdf", pagesize=(470, 188))
    for panel, gamma in enumerate((0.5, 1.0, 2.0)):
        left = 49 + 145 * panel
        pdf.setFont("Vera-Bold", 10)
        pdf.drawString(left, 171, f"({chr(65 + panel)}) Aspect ratio {gamma:g}")
        rows = sorted((row for row in benchmark_rows if row[0] == gamma), key=lambda row: row[1])
        log_snr = np.log10([row[1] for row in rows])
        point = axes(pdf, left, 51, 111, 105, log_snr,
                     [0.8, 1.2, 1.6, 2, 2.4, 2.8] if panel == 0 else (
                         [0.9, 1, 1.1, 1.2] if panel == 1 else [0.8, 1.2, 1.6, 2]),
                     "SNR (log scale)", "Risk / oracle ridge" if panel == 0 else "",
                     x_tick_labels=[str(row[1]) for row in rows])
        series(pdf, point, [(x, row[3] / row[4]) for x, row in zip(log_snr, rows)], COLORS[0])
        if gamma != 1:
            series(pdf, point, [(x, row[5] / row[4]) for x, row in zip(log_snr, rows)], COLORS[2], dash=LINE_STYLES[2])
        series(pdf, point, [(log_snr[0], 1), (log_snr[-1], 1)], COLORS[1], dash=LINE_STYLES[1])
        for index, color in enumerate(COLORS, start=6):
            if index == 8 and gamma == 1:
                continue
            for x, row in zip(log_snr, rows):
                series(pdf, point, [(x, row[index] / row[4])], color, marker=MARKERS[index - 6],
                       errors=[row[index+3] / row[4]])
    pdf.save()

    pdf = figure_canvas(folder / "validation.pdf", pagesize=(470, 203))
    validation_rows = sorted(validation_rows)
    sizes = [r[0] for r in validation_rows]
    columns = ((2, 5), (3, 6), (4, 7), (11, 12))
    excursion = max(abs(row[i]) + 2 * row[j] for row in validation_rows for i, j in columns)
    bound = max(0.04, np.ceil(excursion / 0.02) * 0.02)
    spacing = 0.05 * min(np.diff(sizes))
    limits = (min(sizes) - 2 * spacing, max(sizes) + 2 * spacing)
    point = axes(pdf, 65, 55, 375, 126, sizes, [-bound, -bound/2, 0, bound/2, bound],
                 "Sample size", "Risk or score: two minus one", x_limits=limits)
    series(pdf, point, [(limits[0], 0), (limits[1], 0)], "#808080", dash=(3, 2))
    for offset, ((score, se), (color, dash, marker)) in enumerate(zip(columns, VALIDATION_STYLES)):
        series(pdf, point, [(row[0] + (offset - 1.5) * spacing, row[score]) for row in validation_rows],
               color, errors=[row[se] for row in validation_rows], dash=dash, marker=marker)
    pdf.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test to a separate folder.")
    args = parser.parse_args()
    results = ROOT / ("Code/quick-results" if args.quick else "Code/results")
    plots = results if args.quick else ROOT / "Figures"
    plots.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    n, bench_reps, sizes, val_reps = (
        (48, 3, [100, 200, 300], 3) if args.quick else (240, 80, [200, 700, 1800, 400, 1000, 1400], 100)
    )
    rng = np.random.default_rng(20261119)
    benchmark_rows = benchmark(rng, bench_reps, n)
    validation_rows = sorted(validation(rng, val_reps, sizes))
    if not args.quick:
        # Keep the established Monte Carlo draws at the original settings.
        additional_snrs = [snr for snr in BENCHMARK_SNRS if snr not in (1, 10, 40)]
        benchmark_rows += benchmark(np.random.default_rng(20261120), bench_reps, n, additional_snrs)
        benchmark_rows.sort()
    write_csv(results, "benchmark.csv",
              ["gamma", "snr", "count", "pls_limit", "ridge_limit", "least_squares_limit",
               "pls_mean", "ridge_mean", "least_squares_mean",
               "pls_se", "ridge_se", "least_squares_se",
               "pls_minus_ridge_mean", "pls_minus_least_squares_mean",
               "pls_minus_ridge_se", "pls_minus_least_squares_se"], benchmark_rows)
    write_csv(results, "validation.csv",
              ["n", "holdout_size", "full_gap", "holdout_gap", "five_fold_gap", "full_se",
               "holdout_se", "five_fold_se", "full_two_fraction", "holdout_two_fraction",
               "five_fold_two_fraction", "loo_gap", "loo_se", "loo_two_fraction"], validation_rows)
    cases = []
    for gamma in (0.9, 1.125):
        risk, _, _ = isotropic(gamma, 1.5, count=2)
        cases.append((gamma, *risk))
    write_csv(results, "five-fold-limits.csv", ["gamma", "one_component", "two_components"], cases)
    figures(plots, benchmark_rows, validation_rows)


if __name__ == "__main__":
    main()
