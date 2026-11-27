"""Evaluate plug-in dimension selection on nested isotropic Gaussian designs."""
from math import ceil, log, sqrt

from interpolation_peak import critical_risk
from reproduce import ROOT, np, write_csv


GAMMAS = (0.5, 2.0)
CORRELATIONS_SQUARED = ((0.4, 0.55), (0.4, 0.8))
SEED = 704


def response_scores(gamma, rho, max_count):
    """Predict a new response with unit variance, allowing for omitted signal."""
    snr = rho / (1 - rho)
    return (1 - rho) * np.array([
        1 + critical_risk(gamma, snr, k) for k in range(1, max_count + 1)
    ])


def oracle(gamma, rho):
    """Certify the component optimum and bound the dimension's score by ridge."""
    max_count = 512
    snr = rho / (1 - rho)
    scores = response_scores(gamma, rho, max_count)
    best = int(scores.argmin())

    # On the positive MP support [a,b], the squared residual of PLS is at
    # most (s+gamma)/gamma * (1-a/b)^(2k), by the training trial (1-t/b)^k.
    # The bias is at least the OLS null-space bias. The possible variance
    # saving is at most (2 gamma/a) times the L2 norm of the residual.
    a, b = (1 - sqrt(gamma))**2, (1 + sqrt(gamma))**2
    ols = (gamma / (1 - gamma) if gamma < 1 else
           snr * (1 - 1 / gamma) + 1 / (gamma - 1))
    bound = 2 * gamma / a * sqrt((snr + gamma) / gamma) * (1 - a / b)**max_count
    assert scores[best] < (1 - rho) * (1 + ols - bound)

    penalty = gamma / snr
    linear = penalty + 1 - gamma
    ridge = 2 * gamma / (linear + sqrt(linear**2 + 4 * gamma * penalty))
    return scores[best], best + 1, (1 - rho) * (1 + ridge)


def estimate_rho(projections, diagonal_terms, second_moment, p, n):
    """Off-diagonal U-statistic divided by the sample response second moment."""
    signal = (projections[:p] @ projections[:p] - diagonal_terms) / (n * (n - 1))
    return float(np.clip(signal / second_moment, 1 / n, 1 - 1 / n))


def main():
    rng = np.random.default_rng(SEED)
    repetitions = 200
    rows = []
    oracle_pairs = []
    for corrs in CORRELATIONS_SQUARED:
        pair = [oracle(g, rho) for g, rho in zip(GAMMAS, corrs)]
        winner = int(np.argmin([result[0] for result in pair]))
        assert pair[winner][0] < pair[1 - winner][2]
        oracle_pairs.append((pair, winner))

    for n in (150, 600):
        p_small, p_large = [round(n * gamma) for gamma in GAMMAS]
        max_count = ceil(log(n + 1))
        chosen = np.zeros(len(CORRELATIONS_SQUARED), dtype=int)
        for _ in range(repetitions):
            x = rng.standard_normal((n, p_large))
            noise = rng.standard_normal(n)
            norm_small = np.einsum("ij,ij->i", x[:, :p_small], x[:, :p_small])
            norm_large = norm_small + np.einsum("ij,ij->i", x[:, p_small:], x[:, p_small:])
            for index, (rho_small, rho_large) in enumerate(CORRELATIONS_SQUARED):
                y = (sqrt(rho_small) * x[:, 0] + sqrt(rho_large - rho_small) * x[:, p_small]
                     + sqrt(1 - rho_large) * noise)
                squared_y = y**2
                projections = x.T @ y
                candidate_scores = []
                for gamma, p, norm in zip(GAMMAS, (p_small, p_large), (norm_small, norm_large)):
                    rho = estimate_rho(projections, squared_y @ norm, squared_y.mean(), p, n)
                    candidate_scores.append(response_scores(gamma, rho, max_count).min())
                # A tie selects the smaller dimension.
                chosen[index] += int(candidate_scores[1] < candidate_scores[0])

        for corrs, (pair, winner), count in zip(CORRELATIONS_SQUARED, oracle_pairs, chosen):
            correct = int(count if winner == 1 else repetitions - count)
            frequency = correct / repetitions
            mcse = sqrt(frequency * (1 - frequency) / repetitions)
            rows.append((n, max_count, *corrs, pair[0][0], pair[0][1], pair[0][2],
                         pair[1][0], pair[1][1], pair[1][2], GAMMAS[winner],
                         repetitions, correct, frequency, mcse))
    write_csv(ROOT / "Code/results", "dimension-selection.csv",
              ["n", "max_components", "rho_small", "rho_large", "score_small",
               "optimal_components_small", "ridge_lower_small", "score_large",
               "optimal_components_large", "ridge_lower_large", "oracle_gamma",
               "replications", "correct_count", "correct_fraction", "monte_carlo_se"], rows)


if __name__ == "__main__":
    main()
