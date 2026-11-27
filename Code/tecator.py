"""Evaluate the prescribed Tecator split with and without repeated spectra."""
import argparse
import re
from pathlib import Path

from reproduce import ROOT, np, write_csv

N_TRAIN = 172
N_TEST = 43
COUNTS = np.arange(1, 31)
PENALTIES = np.logspace(-10, 2, 121)
SEED = 20261119
REPEATS = 100


def load_data(path):
    text = path.read_text(encoding="utf-8")
    markers = list(re.finditer(r"(?m)^extrapolation_examples=25[ \t\r]*$", text))
    if not markers:
        raise ValueError("The StatLib data header is missing.")
    records = np.fromstring(text[markers[-1].end():], sep=" ").reshape(240, 125)
    x, y = records[:215, :100], records[:215, 123]
    if x.shape != (215, 100) or y.shape != (215,) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Expected 215 complete spectra and fat percentages.")
    return x, y


def standardise(x, y, new_x):
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    mean_y = y.mean()
    return (x - center) / scale, y - mean_y, (new_x - center) / scale, mean_y


def unique_split(spectra):
    """Keep the first official row for each distinct vector of 100 absorbances."""
    _, first = np.unique(spectra, axis=0, return_index=True)
    first.sort()
    return first[first < N_TRAIN], first[first >= N_TRAIN]


def pls_predictions(x, y, new_x):
    """Fit least squares on an explicitly orthogonalised Krylov basis."""
    x, y, new_x, mean_y = standardise(x, y, new_x)
    gram = x.T @ x / len(y)
    rhs = x.T @ y / len(y)
    basis = np.empty((x.shape[1], len(COUNTS)))
    direction = rhs / np.linalg.norm(rhs)
    predictions = []
    for j in range(len(COUNTS)):
        basis[:, j] = direction
        span = basis[:, :j + 1]
        # Reorthogonalisation handles the strong collinearity of adjacent wavelengths.
        score = x @ span
        coefficient, *_ = np.linalg.lstsq(score, y, rcond=None)
        predictions.append(mean_y + (new_x @ span) @ coefficient)
        if j + 1 < len(COUNTS):
            direction = gram @ direction
            for _ in range(2):
                direction -= span @ (span.T @ direction)
            norm = np.linalg.norm(direction)
            if norm <= 100 * np.finfo(float).eps * np.linalg.norm(gram, ord=np.inf):
                raise ValueError("Krylov subspace stopped growing before the end of the grid.")
            direction /= norm
    return np.asarray(predictions)


def ridge_predictions(x, y, new_x):
    x, y, new_x, mean_y = standardise(x, y, new_x)
    left, singular, right_t = np.linalg.svd(x, full_matrices=False)
    shrinkage = singular[:, None] / (singular[:, None]**2 + len(y) * PENALTIES)
    weights = (left.T @ y)[:, None] * shrinkage
    return mean_y + ((new_x @ right_t.T) @ weights).T


def tune(x, y, folds, predict):
    indices = np.arange(len(y))
    error = None
    size = 0
    for test in folds:
        train = np.setdiff1d(indices, test, assume_unique=True)
        squared = (predict(x[train], y[train], x[test]) - y[test])**2
        batch = squared.sum(axis=1)
        error = batch if error is None else error + batch
        size += len(test)
    return error / size


def evaluate(train_x, train_y, test_x, test_y, test_rows, output, prefix):
    """Tune on these development spectra and evaluate once on the test rows."""
    n_train, n_test = len(train_y), len(test_y)
    rng = np.random.default_rng(SEED)
    fixed = np.array_split(rng.permutation(n_train), 5)
    size = int(np.sqrt(n_train))
    small = [rng.choice(n_train, size, replace=False) for _ in range(REPEATS)]
    leave_one_out = np.arange(n_train).reshape(-1, 1)
    procedures = [
        ("PLS fixed five-fold", pls_predictions, COUNTS, fixed),
        ("PLS repeated small holdout", pls_predictions, COUNTS, small),
        ("Ridge fixed five-fold", ridge_predictions, PENALTIES, fixed),
        ("PLS leave-one-out", pls_predictions, COUNTS, leave_one_out),
    ]
    scores = [tune(train_x, train_y, folds, predict) for _, predict, _, folds in procedures]
    # Hyperparameters are fixed before evaluating any test errors.
    selected = [int(values.argmin()) for values in scores]
    full_paths = {
        pls_predictions: pls_predictions(train_x, train_y, test_x),
        ridge_predictions: ridge_predictions(train_x, train_y, test_x),
    }
    predictions = np.array(
        [full_paths[predict][index] for (_, predict, _, _), index in zip(procedures, selected)]
    )
    errors = predictions - test_y
    methods = [name for name, _, _, _ in procedures]
    write_csv(output, f"{prefix}-test.csv",
              ["method", "components", "ridge_penalty", "validation_rmse", "test_rmse", "test_mae"],
              [(name, int(grid[j]) if predict is pls_predictions else "",
                grid[j] if predict is ridge_predictions else "",
                np.sqrt(scores[i][j]), np.sqrt(np.mean(errors[i]**2)), np.mean(np.abs(errors[i])))
               for i, ((name, predict, grid, _), j) in enumerate(zip(procedures, selected))])
    write_csv(output, f"{prefix}-tuning.csv", ["method", "parameter", "validation_mse"],
              [(name, parameter, error) for (name, _, grid, _), values in zip(procedures, scores)
               for parameter, error in zip(grid, values)])
    write_csv(output, f"{prefix}-predictions.csv", ["StatLib_row", "fat_percent", *methods],
              [(row, actual, *fits) for row, actual, fits in zip(test_rows, test_y, predictions.T)])
    bootstrap = rng.integers(n_test, size=(10000, n_test))
    boot_rmse = np.sqrt(np.mean(errors[:, bootstrap]**2, axis=2))
    rmse = np.sqrt(np.mean(errors**2, axis=1))
    write_csv(output, f"{prefix}-paired-comparisons.csv",
              ["method", "reference", "rmse_difference", "lower_95", "upper_95"],
              [(methods[i], methods[j], rmse[i]-rmse[j],
                *np.quantile(boot_rmse[i]-boot_rmse[j], [0.025, 0.975]))
               for i in range(len(methods)) for j in range(i)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "Code/data/tecator.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "Code/results")
    args = parser.parse_args()
    x, y = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    test_rows = np.arange(N_TRAIN + 1, N_TRAIN + N_TEST + 1)
    evaluate(x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:], y[N_TRAIN:],
             test_rows, args.output, "tecator")
    train, test = unique_split(x)
    evaluate(x[train], y[train], x[test], y[test], test + 1, args.output, "tecator-unique")


if __name__ == "__main__":
    main()
