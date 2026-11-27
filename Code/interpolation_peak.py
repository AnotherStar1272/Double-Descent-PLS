"""Compare risk peaks on fixed-signal and nested Gaussian dimension paths."""
from reproduce import (COLORS, LINE_STYLES, MARKERS, ROOT, axes, figure_canvas,
                       isotropic, np, pls, series, summary, write_csv)


def sinh_ratio(j, end, angle):
    """Evaluate sinh(j*angle)/sinh(end*angle) without overflow."""
    if angle == 0:
        return j / end
    return np.exp((j - end) * angle) * (-np.expm1(-2 * j * angle)) / (
        -np.expm1(-2 * end * angle)
    )


def critical_risk(gamma, snr, count):
    """Compute excess risk in units of the noise variance via the Chebyshev coefficients."""
    end = count + 1
    j = np.arange(1, end)
    scale = gamma**0.5
    angle = abs(np.log(gamma)) / 2
    v = sinh_ratio(end - j, end, angle) / scale
    if snr == 0:
        return gamma * (v @ v)
    delta = gamma / snr
    shifted = np.arccosh((1 + gamma + delta) / (2 * scale))
    w = (sinh_ratio(j, end, angle) - sinh_ratio(j, end, shifted)) / (scale * delta)
    alpha = gamma * v[-1] / (1 + gamma * w[-1])
    u = v - alpha * w
    boundary = np.concatenate(([1 / scale], u, [0]))
    bias = snr * np.sum((scale * boundary[:-1] - boundary[1:])**2)
    variance = gamma * (u @ u)
    return bias + variance


def peak_profile(local_coordinate):
    """Return the limiting rescaled risk at the given local coordinate."""
    x = abs(local_coordinate)
    if x < 1e-3:
        return 1 / 3 - 2 * x**2 / 45 + 2 * x**4 / 315
    return (np.sinh(2 * x) - 2 * x) / (4 * x * np.sinh(x)**2)


def simulate_peak(snr, counts, gammas=(0.8, 1.0, 1.2, 0.7, 0.9, 1.1, 1.3), seed=701):
    """Monte Carlo means and standard errors of the conditional-mean test risk."""
    n = 600
    rng = np.random.default_rng(seed)
    rows = []
    for gamma in gammas:
        p = round(n * gamma)
        beta = np.zeros(p)
        beta[0] = snr**0.5
        risks = []
        for _ in range(80):
            x = rng.standard_normal((n, p))
            y = beta[0] * x[:, 0] + rng.standard_normal(n)
            path = pls(x, y, max(counts))
            risks.append([np.sum((estimate - beta)**2) for k, estimate in enumerate(path, 1)
                          if k in counts])
        means, errors = summary(risks)
        rows.extend((k, gamma, mean, error) for k, mean, error in zip(counts, means, errors))
    return rows


def simulate_nested():
    """Use the same Gaussian design and noise for every nested fit in a replication."""
    n, reps = 600, 80
    gammas = (0.05, 0.1, 1.0, 4.0)
    dimensions = [round(n * gamma) for gamma in gammas]
    signal_column = dimensions[0]  # first column added at the second dimension
    snrs, counts = (0.1, 1.5), (1, 16)
    risks = np.empty((reps, len(snrs), len(counts), len(gammas)))
    rng = np.random.default_rng(702)
    for run in range(reps):
        x = rng.standard_normal((n, dimensions[-1]))
        noise = rng.standard_normal(n)
        for h, snr in enumerate(snrs):
            signal = snr**0.5
            y = x[:, signal_column] * signal + noise
            for d, p in enumerate(dimensions):
                path = pls(x[:, :p], y, max(counts))
                for k, fit in enumerate(path, 1):
                    if k in counts:
                        # Omitted signal contributes its full squared norm.
                        risk = fit @ fit + snr
                        if signal_column < p:
                            risk -= 2 * signal * fit[signal_column]
                        risks[run, h, counts.index(k), d] = risk

    rows = []
    for h, snr in enumerate(snrs):
        for j, k in enumerate(counts):
            path = risks[:, h, j, :]
            means, errors = summary(path)
            changes, change_errors = summary(np.diff(path, axis=1))
            limits = [snr + (1 + snr) * critical_risk(gammas[0], 0, k),
                      *[critical_risk(gamma, snr, k) for gamma in gammas[1:]]]
            for d, gamma in enumerate(gammas):
                rows.append((snr, k, gamma, limits[d], means[d], errors[d],
                             changes[d - 1] if d else "",
                             change_errors[d - 1] if d else ""))
    return rows


def simulate_profile(snr, counts):
    """Measure finite-sample risk across common rescaled locations."""
    rng = np.random.default_rng(703)
    n, reps = 600, 80
    settings = [(k, round(n * np.exp(2 * x / (k + 1))))
                for k in counts for x in (-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2)]
    risks = np.empty((reps, len(settings)))
    for run in range(reps):
        x = rng.standard_normal((n, max(p for _, p in settings)))
        y = snr**0.5 * x[:, 0] + rng.standard_normal(n)
        for j, (k, p) in enumerate(settings):
            fit = list(pls(x[:, :p], y, k))[-1]
            fit[0] -= snr**0.5
            risks[run, j] = fit @ fit
    means, errors = summary(risks)
    return [(k, p / n, mean, se) for (k, p), mean, se in zip(settings, means, errors)]


def main():
    snr = 1.5
    counts = (1, 8, 32)
    for gamma in (0.65, 0.9, 1, 1.12, 1.35):
        path, _, _ = isotropic(gamma, snr, count=max(counts))
        np.testing.assert_allclose([critical_risk(gamma, snr, k) for k in counts],
                                   path[np.asarray(counts) - 1], rtol=1e-5)

    csv_rows = []
    nested = simulate_nested()
    write_csv(ROOT / "Code/results", "nested-peak.csv",
              ["total_snr", "components", "aspect_ratio", "full_mean_risk_limit",
               "finite_sample_mean", "monte_carlo_se", "change_from_previous",
               "paired_change_se"], nested)
    figures = ROOT / "Figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure = figure_canvas(figures / "interpolation-peak.pdf", pagesize=(470, 170))
    for left, label in [(51, "(A)  One component"), (203, "(B)  Component count"),
                        (355, "(C)  Peak profile")]:
        figure.setFont("Vera-Bold", 9)
        figure.drawString(left, 149, label)

    point_weak = axes(figure, 51, 42, 99, 91, [0, 1, 2, 3, 4],
                      [0, 0.1, 0.2, 0.3], "Aspect ratio", "Risk / noise variance",
                      x_limits=(0, 4.1))
    weak_grid = np.linspace(0.05, 4, 201)
    weak_risks = [critical_risk(gamma, 0.1, 1) for gamma in weak_grid]
    series(figure, point_weak, list(zip(weak_grid, weak_risks)), COLORS[0])
    csv_rows.extend(("A", 1, gamma, "", risk, "", "", "")
                    for gamma, risk in zip(weak_grid, weak_risks))
    for s, k, gamma, _, mean, se, _, _ in nested:
        if s == 0.1 and k == 1 and gamma >= 0.1:
            series(figure, point_weak, [(gamma, mean)], COLORS[0], errors=[se], marker=MARKERS[0])
            csv_rows.append(("A_simulation", k, gamma, "", "", "", mean, se))

    weak_gammas = (0.25, 0.55, 0.8, 1.2, 1.6, 2.5, 3.2)
    for k, gamma, mean, se in simulate_peak(0.1, (1,), weak_gammas, seed=705):
        series(figure, point_weak, [(gamma, mean)], COLORS[0], errors=[se], marker=MARKERS[0])
        csv_rows.append(("A_simulation", k, gamma, "", "", "", mean, se))

    aspect_ratios = np.linspace(0.65, 1.35, 281)
    point_b = axes(figure, 203, 42, 99, 91, [0.7, 1, 1.3],
                   [0, 3, 6, 9, 12], "Aspect ratio", "Risk / noise variance",
                   x_limits=(0.65, 1.35))
    series(figure, point_b, [(1, 0), (1, 12)], "#aaaaaa", dash=(2, 3))
    for k, color, dash in zip(counts, COLORS, LINE_STYLES):
        risks = [critical_risk(gamma, snr, k) for gamma in aspect_ratios]
        series(figure, point_b, list(zip(aspect_ratios, risks)), color, dash=dash)
        csv_rows.extend(("B", k, gamma, "", risk, "", "", "")
                        for gamma, risk in zip(aspect_ratios, risks))

    near_one = simulate_peak(snr, counts)
    for k, gamma, mean, se in near_one:
        index = counts.index(k)
        series(figure, point_b, [(gamma, mean)], COLORS[index], errors=[se], marker=MARKERS[index])
        csv_rows.append(("B_simulation", k, gamma, "", "", "", mean, se))

    coords = np.linspace(-5, 5, 251)
    point_c = axes(figure, 355, 42, 99, 91, [-5, 0, 5],
                   [0, 0.1, 0.2, 0.3, 0.4], "Local coordinate", "Rescaled risk")
    for k, color, dash in zip(counts[1:], COLORS[1:], LINE_STYLES[1:]):
        ratios = np.exp(2 * coords / (k + 1))
        risks = [critical_risk(gamma, snr, k) for gamma in ratios]
        scaled = np.asarray(risks) / (k + 1)
        series(figure, point_c, list(zip(coords, scaled)), color, dash=dash)
        csv_rows.extend(("C", k, gamma, x, risk, value, "", "")
                        for gamma, x, risk, value in zip(ratios, coords, risks, scaled))
    limit = [peak_profile(x) for x in coords]
    series(figure, point_c, list(zip(coords, limit)), COLORS[0])
    csv_rows.extend(("C", "limit", "", x, "", value, "", "") for x, value in zip(coords, limit))
    centres = [row for row in near_one if row[1] == 1 and row[0] in counts[1:]]
    for k, gamma, mean, se in centres + simulate_profile(snr, counts[1:]):
        index = counts.index(k)
        x = (k + 1) * np.log(gamma) / 2
        series(figure, point_c, [(x, mean / (k + 1))], COLORS[index],
               errors=[se / (k + 1)], marker=MARKERS[index])
        csv_rows.append(("C_simulation", k, gamma, x, "", "", mean, se))
    figure.save()

    write_csv(ROOT / "Code/results", "interpolation-peak.csv",
              ["panel", "components", "aspect_ratio", "local_coordinate",
               "risk_over_noise_variance", "risk_over_noise_variance_and_k_plus_1",
               "finite_sample_mean", "monte_carlo_se"],
              csv_rows)


if __name__ == "__main__":
    main()
