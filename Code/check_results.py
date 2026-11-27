"""Check the core numerical calculations independently of the plotting code."""
import csv
from fractions import Fraction
from itertools import combinations, product
from math import ceil, comb, log, prod

from dimension_selection import estimate_rho
from interpolation_peak import critical_risk, peak_profile
from reproduce import BENCHMARK_SNRS, ROOT, isotropic, loo_two_count_predictions, np, pls
from tecator import (COUNTS, N_TEST, N_TRAIN, PENALTIES, REPEATS, SEED, load_data,
                     ridge_predictions, standardise, unique_split)


def exact_pls_risk(gamma, snr, count):
    """Solve the small polynomial normal equations with rational MP moments."""
    moments = [Fraction(1)]
    for degree in range(1, 2 * count + 2):
        moments.append(sum(comb(degree, j) * comb(degree, j + 1) * gamma**j
                           for j in range(degree)) / degree)
    a = [snr * moments[i + 1] + gamma * moments[i] for i in range(2 * count + 1)]
    system = [[a[i + j + 2] for j in range(count)] + [a[i + 1]] for i in range(count)]
    # The normal matrix is positive definite, so all elimination pivots are positive.
    for i in range(count):
        pivot = system[i][i]
        assert pivot > 0
        system[i] = [v / pivot for v in system[i]]
        for j in range(count):
            if j != i:
                factor = system[j][i]
                system[j] = [v - factor * w for v, w in zip(system[j], system[i])]
    coefficients = [row[-1] for row in system]
    linear = sum(c * moments[i + 1] for i, c in enumerate(coefficients))
    quadratic = sum(ci * cj * a[i + j + 1] for i, ci in enumerate(coefficients)
                    for j, cj in enumerate(coefficients))
    return snr - 2 * snr * linear + quadratic


def check_counterexample():
    snr = Fraction(3, 2)
    bounds = [(Fraction(-18, 1000), Fraction(-17, 1000)),
              (Fraction(12, 1000), Fraction(13, 1000))]
    with (ROOT / "Code/results/five-fold-limits.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(bounds)
    for row, (lower, upper) in zip(rows, bounds):
        gamma = Fraction(row["gamma"])
        one, two = (exact_pls_risk(gamma, snr, k) for k in (1, 2))
        assert lower < two - one < upper
        np.testing.assert_allclose(
            [float(row["one_component"]), float(row["two_components"])],
            [float(one), float(two)], atol=1e-12, rtol=0,
        )

    # Verify the polynomial expression for the risk gap with exact arithmetic.
    coefficients = (-2187, -25515, -128790, -365472, -567027, -356841,
                    267312, 628410, 374875, 84375)
    for gamma in (Fraction(i, 4) for i in range(1, 18)):
        polynomial = sum(c * gamma**j for j, c in enumerate(coefficients))
        a2 = 3 + 11 * gamma + 5 * gamma**2
        det_over_gamma = 9 + 39 * gamma + 88 * gamma**2 + 61 * gamma**3 + 25 * gamma**4
        one, two = (exact_pls_risk(gamma, snr, k) for k in (1, 2))
        assert two - one == gamma * polynomial / (2 * (a2 * det_over_gamma)**2)

    assert exact_pls_risk(Fraction(1, 4), 1, 1) == Fraction(97, 289)
    assert exact_pls_risk(Fraction(1, 4), 1, 2) == Fraction(2905, 10201)


def check_simulated_loo():
    rng = np.random.default_rng(595)
    for n, p in [(15, 9), (14, 14), (12, 19)]:
        x = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        all_rows = np.arange(n)
        direct = np.asarray([
            [x[i] @ b for b in pls(x[all_rows != i], y[all_rows != i], 2)]
            for i in all_rows
        ])
        np.testing.assert_allclose(loo_two_count_predictions(x, y), direct, atol=1e-10, rtol=1e-10)


def check_estimators():
    rng = np.random.default_rng(731)
    for n, p in [(40, 26), (28, 45)]:
        x = rng.normal(size=(n, p))
        y = rng.normal(size=n)
        column = x.T @ y / n
        gram = x.T @ x / n
        basis = []
        for estimate in pls(x, y, 5):
            basis.append(column)
            krylov = np.column_stack(basis)
            orthogonal, _ = np.linalg.qr(krylov)
            coefficients, *_ = np.linalg.lstsq(x @ orthogonal, y, rcond=None)
            np.testing.assert_allclose(estimate, orthogonal @ coefficients,
                                       atol=1e-10, rtol=1e-10)
            column = gram @ column


def check_benchmarks():
    with (ROOT / "Code/results/benchmark.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    expected = {(gamma, snr) for gamma in (0.5, 1.0, 2.0) for snr in BENCHMARK_SNRS}
    assert len(rows) == len(expected)
    assert {(float(row["gamma"]), float(row["snr"])) for row in rows} == expected
    for row in rows:
        gamma, snr = float(row["gamma"]), float(row["snr"])
        path, ridge, ols = isotropic(gamma, snr, nodes=2048)
        assert path.argmin() + 1 == int(row["count"])
        np.testing.assert_allclose(path.min(), float(row["pls_limit"]), atol=1e-10)
        penalty = gamma / snr
        # Positive root of the Marchenko–Pastur Stieltjes equation at -penalty.
        linear = penalty + 1 - gamma
        transform = 2 / (np.sqrt(linear**2 + 4 * gamma * penalty) + linear)
        np.testing.assert_allclose(ridge, gamma * transform, atol=1e-10)
        assert ridge < path.min() < ols
        for method in ("ridge", "least_squares"):
            gap = float(row[f"pls_minus_{method}_mean"])
            error = float(row[f"pls_minus_{method}_se"])
            if method == "least_squares" and gamma == 1:
                assert np.isnan(gap) and np.isnan(error)
                continue
            np.testing.assert_allclose(gap, float(row["pls_mean"]) - float(row[f"{method}_mean"]),
                                       atol=1e-12)
            assert 0 < error <= float(row["pls_se"]) + float(row[f"{method}_se"])


def noncrossing_partitions(positions):
    """Choose the block of the first position, then partition each gap."""
    if not positions:
        yield ()
        return
    for size in range(len(positions)):
        for others in combinations(range(1, len(positions)), size):
            indices = (0, *others)
            first = tuple(positions[j] for j in indices)
            ends = (*others, len(positions))
            gaps = [positions[i + 1:j] for i, j in zip(indices, ends)]
            for choices in product(*(noncrossing_partitions(gap) for gap in gaps)):
                yield (first,) + tuple(block for gap in choices for block in gap)


def check_aligned_risks():
    """Compute the two-component risks exactly and optimise ridge over every penalty."""
    a, b = Fraction(7, 10), Fraction(11, 5)
    moments = [(4 * a**q + b**q) / 5 for q in range(6)]

    def sample_moment(q, signal=None, mark=None):
        # At unit aspect ratio every noncrossing partition has weight one.
        # Marking a position inserts one population covariance; anchoring
        # the first block at the signal weights it by that eigenvalue.
        total = 0
        for blocks in noncrossing_partitions(range(q)):
            factors = []
            for block in blocks:
                degree = len(block) + (mark in block)
                weight = signal**degree if signal is not None and 0 in block else moments[degree]
                factors.append(weight)
            total += prod(factors)
        return total

    for t, h, table_pls, table_ridge in ((a, b, 3.0201, 2.1165), (b, a, 0.8162, 0.8839)):
        norm_squared = 5 / t
        # Normal equations on the two Krylov columns (g, Sg).
        s0, s1, s2, s3 = [norm_squared * sample_moment(j + 2, signal=t) + sample_moment(j + 1)
                          for j in range(4)]
        det = s1 * s3 - s2**2
        assert det > 0
        coefficients = [(s0 * s3 - s1 * s2) / det, (s1**2 - s0 * s2) / det]
        linear = sum(c * norm_squared * t * sample_moment(i, signal=t)
                     for i, c in enumerate(coefficients, 1))
        quadratic = sum(ci * cj * (norm_squared * sample_moment(i + j, signal=t, mark=i)
                                   + sample_moment(i + j - 1, mark=0))
                        for i, ci in enumerate(coefficients, 1)
                        for j, cj in enumerate(coefficients, 1))
        pls_risk = 5 - 2 * linear + quadratic
        assert round(float(pls_risk), 4) == table_pls

        # In the effective penalty u, risk is N(u)/D(u). Use exact polynomial
        # coefficients until the root solve to avoid a spurious leading term
        # in N'D - ND'. At zero the risk diverges and at infinity it is five.
        u = np.polynomial.Polynomial([Fraction(0), Fraction(1)])
        q = (u + a)**2 * (u + b)**2
        variance = (4 * a**2 * (u + b)**2 + b**2 * (u + a)**2) / 5
        numerator = 5 * u**2 * (u + h)**2 + variance
        denominator = q - variance
        def derivative(poly):
            return np.polynomial.Polynomial([i * c for i, c in enumerate(poly.coef) if i])

        stationary = (derivative(numerator) * denominator - numerator * derivative(denominator)).trim()
        # Both exact coefficient lists have a single sign change. Descartes'
        # rule therefore certifies that the root search exhausts the positive axis.
        signs = [c > 0 for c in stationary.coef if c]
        assert not signs[0] and signs[-1] and sum(i != j for i, j in zip(signs, signs[1:])) == 1
        roots = np.polynomial.polynomial.polyroots(np.asarray(stationary.coef, dtype=float))
        critical = [z.real for z in roots if abs(z.imag) < 1e-8 and z.real > 0]
        assert denominator(0) == 0 < numerator(0) and len(critical) == 1
        endpoints = [np.inf, float(numerator.coef[-1] / denominator.coef[-1])]
        ridge_risk = min(endpoints + [float(numerator(z) / denominator(z)) for z in critical])
        assert round(ridge_risk, 4) == table_ridge
        print(f"Signal eigenvalue {float(t):g}: two-component PLS {float(pls_risk):.6f}, "
              f"optimal ridge {ridge_risk:.6f}")


def check_critical_profiles():
    nodes, weights = np.polynomial.legendre.leggauss(48)
    y, weights = (nodes + 1) / 2, weights / 2

    def high_signal_energy(x):
        a = 1 / (x / np.tanh(x) - 1)
        u = 1 - (1 + a) * y + a * np.sinh(x * y) / np.sinh(x)
        derivative = -(1 + a) + a * x * np.cosh(x * y) / np.sinh(x)
        return weights @ (derivative**2 / x + x * u**2)

    # Psi >= 1/x. For x >= 5, u >= 1/4 on [0,1/2], hence Psi >= x/32.
    # The values outside this interval thus exceed the interior trial value.
    lower, upper = 0.8, 40
    for _ in range(3):
        grid = np.linspace(lower, upper, 1001)
        energy = np.array([high_signal_energy(x) for x in grid])
        best = int(energy.argmin())
        assert 0 < best < len(grid) - 1
        lower, upper = grid[best - 1], grid[best + 1]
    np.testing.assert_allclose(energy[best], 1.07323, rtol=0, atol=0.000005)

    # Integrate Phi independently and check the local rescaling at large k.
    end = 8192
    for x in (-4, -2, -0.5, 0, 0.5, 2, 4):
        profile = y if x == 0 else np.sinh(x * y) / np.sinh(x)
        target = weights @ profile**2
        np.testing.assert_allclose(peak_profile(x), target, rtol=0, atol=1e-12)
        gamma = np.exp(2 * x / end)
        np.testing.assert_allclose(critical_risk(gamma, 1.5, end - 1) / end,
                                   target, rtol=0, atol=0.00015)


def check_dimension_examples():
    snr, gamma = 0.01, 0.5
    a = snr * (1 + gamma) + gamma
    b = snr * (1 + 3 * gamma + gamma**2) + gamma * (1 + gamma)
    lower = max(a**1.5 / b - snr**0.5, 0)**2
    upper = isotropic(100, snr, count=1)[0][0]
    np.testing.assert_allclose([lower, upper], [0.14089, 0.01970], rtol=0, atol=0.000005)
    over_rho = 0.80
    one = isotropic(2, over_rho / (1 - over_rho), count=1)[0][0]
    over_score = (1 - over_rho) * (1 + one)
    np.testing.assert_allclose(over_score, 0.77152, rtol=0, atol=1e-12)
    assert over_score < 1 - 0.01
    under_score = (1 - 0.95) / (1 - 0.5)
    over_lower = 1 - 0.96 / 2
    np.testing.assert_allclose([under_score, over_lower], [0.10, 0.52])


def check_dimension_selection():
    """Verify the dimension experiment with pairwise sums and exact MP risks."""
    # Exercise both coordinate prefixes and both clipping endpoints.
    rng = np.random.default_rng(117)
    n = 8
    y = np.array([1, -1, 2, 1, -2, 1, -1, 2])
    x = rng.normal(scale=0.1, size=(n, 5))
    x[:, 0] = 1 / y
    for scale, endpoint in ((0.05, 1 / n), (0.7, None), (4, 1 - 1 / n)):
        design = scale * x
        projections = design.T @ y
        for p in (2, 5):
            prefix = design[:, :p]
            signal = 2 * sum(y[i] * y[j] * (prefix[i] @ prefix[j])
                             for i in range(n) for j in range(i)) / (n * (n - 1))
            raw_rho = signal / np.mean(y**2)
            expected = max(1 / n, min(raw_rho, 1 - 1 / n))
            diagonal = np.sum((y[:, None] * prefix)**2)
            estimate = estimate_rho(projections, diagonal, np.mean(y**2), p, n)
            np.testing.assert_allclose(estimate, expected, rtol=0, atol=1e-13)
            if endpoint is None:
                assert 1 / n < expected < 1 - 1 / n
            else:
                assert expected == endpoint

    # Compute all small-count risks exactly and certify the rest of the path.
    gammas = (Fraction(1, 2), Fraction(2))
    rho_small = Fraction(2, 5)
    rho_large_values = (Fraction(11, 20), Fraction(4, 5))
    references = {}
    models = [(gammas[0], rho_small)] + [(gammas[1], rho) for rho in rho_large_values]
    for gamma, rho in models:
        snr = rho / (1 - rho)
        risks = [(1 - rho) * (1 + exact_pls_risk(gamma, snr, k)) for k in range(1, 18)]
        best = risks.index(min(risks)) + 1
        # For these two gammas the shifted Chebyshev trial residual has
        # supremum <= 2*(1/2)^(k/2) on the positive MP support [a,b].
        # PLS minimises training loss, hence its positive-support L2 residual
        # norm is <= 2*((snr+gamma)/gamma)^(1/2)/512 for k >= 18.
        # The variance saving relative to OLS is <= 2*gamma/a times this norm.
        a_lower = min(Fraction(1), gamma) / 6
        assert (1 + gamma - a_lower)**2 > 4 * gamma and snr + gamma <= 4 * gamma
        ols = gamma / (1 - gamma) if gamma < 1 else snr * (1 - 1 / gamma) + 1 / (gamma - 1)
        tail_lower = (1 - rho) * (1 + ols - 8 * gamma / (512 * a_lower))
        assert min(risks) < tail_lower and risks.count(min(risks)) == 1
        # Direct spectral quadrature checks the separate closed-form ridge bound.
        _, ridge, _ = isotropic(float(gamma), float(snr), count=1, nodes=1024)
        references[gamma, rho] = float(min(risks)), best, float(1 - rho) * (1 + ridge)

    with (ROOT / "Code/results/dimension-selection.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    cases = [(int(row["n"]), Fraction(row["rho_small"]), Fraction(row["rho_large"]))
             for row in rows]
    assert sorted(cases) == [(n, rho_small, rho) for n in (150, 600) for rho in rho_large_values]
    for row, (n, small, large) in zip(rows, cases):
        assert int(row["max_components"]) == ceil(log(n + 1))
        pair = [references[gammas[0], small], references[gammas[1], large]]
        for label, (score, count, ridge_score) in zip(("small", "large"), pair):
            np.testing.assert_allclose([float(row["score_" + label]), float(row["ridge_lower_" + label])],
                                       [score, ridge_score], rtol=0, atol=1e-12)
            assert int(row["optimal_components_" + label]) == count <= int(row["max_components"])
        winner = int(pair[1][0] < pair[0][0])
        assert pair[winner][0] < pair[1 - winner][2]
        assert Fraction(row["oracle_gamma"]) == gammas[winner]
        reps, correct = int(row["replications"]), int(row["correct_count"])
        assert reps == 200 and 0 <= correct <= reps
        rate = correct / reps
        np.testing.assert_allclose([float(row["correct_fraction"]), float(row["monte_carlo_se"])],
                                   [rate, (rate * (1 - rate) / reps)**0.5], rtol=0, atol=1e-14)


def check_nested_peaks():
    """Compare nested limits with a separate rational-moment solver."""
    with (ROOT / "Code/results/nested-peak.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    for snr in (Fraction(1, 10), Fraction(3, 2)):
        for k in (1, 16):
            group = [row for row in rows if Fraction(row["total_snr"]) == snr
                     and int(row["components"]) == k]
            gammas = [Fraction(row["aspect_ratio"]) for row in group]
            assert gammas == [Fraction(1, 20), Fraction(1, 10), 1, 4]
            exact = [snr + (1 + snr) * exact_pls_risk(gammas[0], 0, k),
                     *[exact_pls_risk(g, snr, k) for g in gammas[1:]]]
            np.testing.assert_allclose([float(row["full_mean_risk_limit"]) for row in group],
                                       [float(risk) for risk in exact], atol=1e-12, rtol=0)
            mean_changes = np.diff([float(row["finite_sample_mean"]) for row in group])
            paired_se = np.asarray([float(row["paired_change_se"]) for row in group[1:]])
            np.testing.assert_allclose(mean_changes,
                                       [float(row["change_from_previous"]) for row in group[1:]])
            exact_direction = np.sign(np.diff([float(risk) for risk in exact]))
            assert np.all(exact_direction * mean_changes > 2 * paired_se)


def deflated_pls_predictions(train_x, train_y, test_x):
    """Independent component path from successive deflation of X and y."""
    center, scale = train_x.mean(axis=0), train_x.std(axis=0)
    x, new_x = (train_x - center) / scale, (test_x - center) / scale
    y = train_y - train_y.mean()
    model = np.full(len(test_x), train_y.mean())
    path = []
    for _ in COUNTS:
        direction = x.T @ y
        direction /= np.linalg.norm(direction)
        score, test_score = x @ direction, new_x @ direction
        norm_squared = score @ score
        x_loading = x.T @ score / norm_squared
        y_loading = y @ score / norm_squared
        model = model + test_score * y_loading
        path.append(model)
        x -= np.outer(score, x_loading)
        y -= score * y_loading
        new_x -= np.outer(test_score, x_loading)
    return np.array(path)


def check_tecator_split(spectra, fat, train, test, prefix, expected_counts):
    def rows(suffix):
        with (ROOT / "Code/results" / f"{prefix}-{suffix}.csv").open(newline="") as data:
            return list(csv.DictReader(data))

    train_x, train_y = spectra[train], fat[train]
    test_x, test_y = spectra[test], fat[test]
    predictions = rows("predictions")
    np.testing.assert_array_equal([int(row["StatLib_row"]) for row in predictions], test + 1)
    np.testing.assert_array_equal([float(row["fat_percent"]) for row in predictions], test_y)
    scores, summaries = rows("tuning"), rows("test")
    n = len(train)
    rng = np.random.default_rng(SEED)
    five = np.array_split(rng.permutation(n), 5)
    small = [rng.choice(n, int(np.sqrt(n)), replace=False) for _ in range(REPEATS)]
    splits = {"PLS fixed five-fold": five, "PLS repeated small holdout": small,
              "Ridge fixed five-fold": five, "PLS leave-one-out": np.arange(n).reshape(-1, 1)}
    assert [row["method"] for row in summaries] == list(splits)
    pls_path = deflated_pls_predictions(train_x, train_y, test_x)
    x, y, new_x, offset = standardise(train_x, train_y, test_x)
    gram, rhs = x.T @ x, x.T @ y
    errors = {}
    for summary in summaries:
        method = summary["method"]
        is_ridge = method == "Ridge fixed five-fold"
        candidates = [row for row in scores if row["method"] == method]
        grid = PENALTIES if is_ridge else COUNTS
        np.testing.assert_array_equal([float(row["parameter"]) for row in candidates], grid)
        total_sse = np.zeros(len(grid))
        total_tested = 0
        for validation in splits[method]:
            keep = np.ones(n, dtype=bool)
            keep[validation] = False
            fit = ridge_predictions if is_ridge else deflated_pls_predictions
            path = fit(train_x[keep], train_y[keep], train_x[validation])
            total_sse += np.sum((path - train_y[validation])**2, axis=1)
            total_tested += len(validation)
        mse = total_sse / total_tested
        recorded = [float(row["validation_mse"]) for row in candidates]
        # Krylov and deflation algorithms diverge slightly at high counts on these collinear data.
        np.testing.assert_allclose(recorded, mse, atol=5e-5, rtol=1e-6)
        best = int(np.argmin(mse))
        assert best == int(np.argmin(recorded))
        selected = float(summary["ridge_penalty"] if is_ridge else summary["components"])
        assert selected == grid[best]
        np.testing.assert_allclose(float(summary["validation_rmse"]), np.sqrt(mse[best]), atol=1e-7)
        if is_ridge:
            # Direct solve checks the selected SVD ridge fit as well as the baseline grid point.
            for index in (25, best):
                direct = offset + new_x @ np.linalg.solve(gram + n * PENALTIES[index] * np.eye(len(rhs)), rhs)
                fit = ridge_predictions(train_x, train_y, test_x)[index]
                np.testing.assert_allclose(fit, direct, atol=1e-6, rtol=1e-7)
            prediction = direct
        else:
            assert selected == expected_counts[method]
            prediction = pls_path[best]
        saved = np.array([float(row[method]) for row in predictions])
        np.testing.assert_allclose(saved, prediction, atol=1e-6, rtol=0)
        errors[method] = saved - test_y
        np.testing.assert_allclose(float(summary["test_rmse"]), np.sqrt(np.mean(errors[method]**2)))
        np.testing.assert_allclose(float(summary["test_mae"]), np.mean(np.abs(errors[method])))

    # Reuse the original test-row resamples for every pair of methods.
    bootstrap = rng.integers(len(test), size=(10000, len(test)))
    paired = rows("paired-comparisons")
    methods = list(splits)
    assert [(row["method"], row["reference"]) for row in paired] == [
        (m, ref) for i, m in enumerate(methods) for ref in methods[:i]]
    for row in paired:
        left, right = errors[row["method"]], errors[row["reference"]]
        gap = np.sqrt(np.mean(left**2)) - np.sqrt(np.mean(right**2))
        distribution = np.sqrt(np.mean(left[bootstrap]**2, axis=1)) - np.sqrt(np.mean(right[bootstrap]**2, axis=1))
        np.testing.assert_allclose([float(row[k]) for k in ("rmse_difference", "lower_95", "upper_95")],
                                   [gap, *np.quantile(distribution, [0.025, 0.975])], atol=1e-12)


def check_tecator():
    spectra, fat = load_data(ROOT / "Code/data/tecator.txt")
    train, test = np.arange(N_TRAIN), np.arange(N_TRAIN, N_TRAIN + N_TEST)
    counts = {"PLS fixed five-fold": 13, "PLS repeated small holdout": 14, "PLS leave-one-out": 14}
    check_tecator_split(spectra, fat, train, test, "tecator", counts)

    # Reconstruct first occurrences directly from wavelengths, independent of the response and sorting.
    first = np.array([i for i in range(len(spectra))
                      if not np.any(np.all(spectra[:i] == spectra[i], axis=1))])
    keep_train, keep_test = unique_split(spectra)
    np.testing.assert_array_equal(keep_train, first[first < N_TRAIN])
    np.testing.assert_array_equal(keep_test, first[first >= N_TRAIN])
    assert len(keep_train) == 158 and len(keep_test) == 35
    assert not set(map(tuple, spectra[keep_train])) & set(map(tuple, spectra[keep_test]))
    check_tecator_split(spectra, fat, keep_train, keep_test, "tecator-unique", dict.fromkeys(counts, 13))


if __name__ == "__main__":
    check_counterexample()
    check_simulated_loo()
    check_estimators()
    check_benchmarks()
    check_aligned_risks()
    check_critical_profiles()
    check_dimension_examples()
    check_dimension_selection()
    check_nested_peaks()
    check_tecator()
    print("Numerical checks passed.")
