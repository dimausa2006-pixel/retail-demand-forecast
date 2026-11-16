"""Compare simple one-week-ahead forecasts of weekly revenue on weeks the methods have not seen.

Usage:
    python src/forecast.py

Reads data/weekly_sales.csv, writes reports/results.md and reports/forecast.png.

How the test works (rolling origin):
    For each of the last TEST_WEEKS weeks, every method sees only the weeks before it,
    makes one forecast for that week, and the forecast is compared with what really happened.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_WEEKS = 26      # the last half year is used for testing
SEASON = 52          # weeks in a year
LAGS = (1, 2, 3, 4, 52)


# ---------- the forecasting methods ----------
# Each takes `history` (a 1-D array of past weekly values, oldest first)
# and returns one number: the forecast for the next week.

def last_week(history):
    """Next week will look like the week that just ended."""
    return float(history[-1])


def four_week_average(history):
    """Average of the last four weeks."""
    return float(np.mean(history[-4:]))


def same_week_last_year(history):
    """Next week will look like the same week one year ago."""
    return float(history[-SEASON])


def last_year_with_growth(history, window=8):
    """Same week last year, scaled by how much the last 8 weeks grew against the same 8 weeks a year ago."""
    recent = history[-window:].sum()
    year_ago = history[-window - SEASON:-SEASON].sum()
    growth = recent / year_ago if year_ago > 0 else 1.0
    return float(history[-SEASON] * growth)


def exponential_smoothing(history, alpha=None):
    """Weighted average of all past weeks, where recent weeks count more.

    alpha (0 to 1) says how fast old weeks are forgotten. If it is not given,
    the value that would have forecast the history best is chosen.
    """
    if alpha is None:
        alpha = best_alpha(history)
    level = history[0]
    for value in history[1:]:
        level = alpha * value + (1 - alpha) * level
    return float(level)


def best_alpha(history, grid=np.arange(0.1, 1.0, 0.1)):
    """Try several alphas on the history and return the one with the smallest average error."""
    best, best_error = grid[0], np.inf
    for alpha in grid:
        level, errors = history[0], []
        for value in history[1:]:
            errors.append(abs(value - level))
            level = alpha * value + (1 - alpha) * level
        if np.mean(errors) < best_error:
            best, best_error = alpha, np.mean(errors)
    return float(best)


def lag_regression(history, ridge=1.0):
    """Linear regression on past values: last 4 weeks and the same week last year.

    Fitted again every week on the history available so far. A small ridge penalty
    keeps the weights stable because there are few training rows.
    """
    rows, targets = [], []
    for t in range(max(LAGS), len(history)):
        rows.append([history[t - lag] for lag in LAGS])
        targets.append(history[t])
    X, y = np.array(rows, dtype=float), np.array(targets, dtype=float)
    # Put every column on the same scale so the penalty treats them equally.
    scale = X.mean(axis=0)
    scale[scale == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), X / scale])
    penalty = ridge * np.eye(Xs.shape[1])
    penalty[0, 0] = 0.0  # do not shrink the constant term
    weights = np.linalg.solve(Xs.T @ Xs + penalty, Xs.T @ y)
    next_row = np.array([history[len(history) - lag] for lag in LAGS], dtype=float) / scale
    return float(weights[0] + next_row @ weights[1:])


METHODS = {
    "Last week": last_week,
    "4-week average": four_week_average,
    "Exponential smoothing": exponential_smoothing,
    "Same week last year": same_week_last_year,
    "Last year x recent growth": last_year_with_growth,
    "Regression on past weeks": lag_regression,
}


# ---------- the test ----------

def rolling_forecasts(values, test_weeks=TEST_WEEKS, methods=METHODS):
    """Return {method name: array of forecasts} for the last `test_weeks` weeks."""
    values = np.asarray(values, dtype=float)
    start = len(values) - test_weeks
    forecasts = {name: [] for name in methods}
    for t in range(start, len(values)):
        history = values[:t]            # only the past is visible
        for name, method in methods.items():
            forecasts[name].append(method(history))
    return {name: np.array(f) for name, f in forecasts.items()}


def score(actual, forecast):
    """MAE: average miss in pounds. WAPE: total miss as a share of total sales. Bias: average over(+)/under(-)."""
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    error = forecast - actual
    return {
        "MAE": float(np.mean(np.abs(error))),
        "WAPE %": float(100 * np.abs(error).sum() / actual.sum()),
        "Bias": float(np.mean(error)),
    }


def main():
    weekly = pd.read_csv(ROOT / "data" / "weekly_sales.csv", parse_dates=["week_start"])
    values = weekly["revenue"].to_numpy()
    test = weekly.iloc[-TEST_WEEKS:]
    actual = test["revenue"].to_numpy()

    forecasts = rolling_forecasts(values)
    table = pd.DataFrame({name: score(actual, f) for name, f in forecasts.items()}).T
    table = table.sort_values("WAPE %")
    best = table.index[0]

    lines = ["# Results", "",
             f"Target: weekly revenue (GBP). {len(weekly)} weeks in total; the last {TEST_WEEKS} "
             f"({test['week_start'].min():%d %b %Y} to {test['week_start'].max():%d %b %Y}) are the test weeks.",
             f"Average revenue in a test week: £{actual.mean():,.0f}.", "",
             "| Method | MAE (£) | WAPE % | Bias (£) |", "|:--|--:|--:|--:|"]
    for name, row in table.iterrows():
        lines.append(f"| {name} | {row['MAE']:,.0f} | {row['WAPE %']:.1f} | {row['Bias']:+,.0f} |")
    lines += ["", f"Best on these weeks: **{best}**."]
    (ROOT / "reports" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).replace("£", "GBP "))

    detail = test[["week_start", "revenue"]].copy()
    for name, f in forecasts.items():
        detail[name] = np.round(f, 0)
    detail.to_csv(ROOT / "reports" / "test_week_forecasts.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(weekly["week_start"], weekly["revenue"] / 1000, color="#444", linewidth=1.3, label="Actual")
    ax.plot(test["week_start"], forecasts[best] / 1000, color="#d1495b", linewidth=1.8, label=f"Forecast: {best}")
    ax.plot(test["week_start"], forecasts["Last week"] / 1000, color="#2e86ab", linewidth=1, linestyle="--",
            label="Forecast: Last week")
    ax.axvline(test["week_start"].iloc[0], color="#999", linestyle=":", linewidth=1)
    ax.set_ylabel("Weekly revenue (thousand GBP)")
    ax.set_title("One-week-ahead forecasts on the last 26 weeks")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "forecast.png", dpi=150)


if __name__ == "__main__":
    main()
