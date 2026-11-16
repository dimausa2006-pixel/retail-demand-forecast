"""Run with:  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import forecast as fc      # noqa: E402
import prepare as pr       # noqa: E402


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.history = np.arange(1.0, 121.0)   # 1, 2, ..., 120

    def test_last_week(self):
        self.assertEqual(fc.last_week(self.history), 120.0)

    def test_four_week_average(self):
        self.assertEqual(fc.four_week_average(self.history), np.mean([117, 118, 119, 120]))

    def test_same_week_last_year(self):
        # the next week is number 121, so one year earlier is number 69
        self.assertEqual(fc.same_week_last_year(self.history), 69.0)

    def test_growth_is_one_when_nothing_changed(self):
        flat = np.tile(np.arange(1.0, 53.0), 3)          # the same year three times
        self.assertAlmostEqual(fc.last_year_with_growth(flat), fc.same_week_last_year(flat))

    def test_exponential_smoothing_on_constant_series(self):
        self.assertAlmostEqual(fc.exponential_smoothing(np.full(30, 7.0)), 7.0)

    def test_regression_learns_a_straight_line(self):
        # each week is one more than the week before, so the next value is 121
        self.assertAlmostEqual(fc.lag_regression(self.history, ridge=0.0), 121.0, places=3)


class RollingTests(unittest.TestCase):
    def test_no_look_ahead(self):
        """Changing a week must not change any forecast made for that week or earlier."""
        rng = np.random.default_rng(0)
        values = rng.uniform(50, 150, size=130)
        before = fc.rolling_forecasts(values, test_weeks=10)
        changed = values.copy()
        changed[-1] = 10_000.0                           # only the very last week changes
        after = fc.rolling_forecasts(changed, test_weeks=10)
        for name in before:
            np.testing.assert_allclose(before[name], after[name])

    def test_score(self):
        s = fc.score([100, 100], [110, 80])
        self.assertEqual(s["MAE"], 15.0)
        self.assertEqual(s["WAPE %"], 15.0)
        self.assertEqual(s["Bias"], -5.0)


class CleaningTests(unittest.TestCase):
    def test_rules_remove_the_right_rows(self):
        raw = pd.DataFrame({
            "Invoice":     ["1", "C2", "3", "4", "5", "1"],
            "StockCode":   ["10001", "10001", "10002", "10003", "POST", "10001"],
            "Quantity":    [2, -2, 0, 5, 1, 2],
            "InvoiceDate": pd.to_datetime(["2010-01-04"] * 6),
            "Price":       [3.0, 3.0, 1.0, 0.0, 9.0, 3.0],
            "sheet":       ["a", "a", "a", "a", "a", "b"],
        })
        clean, log = pr.clean(raw)
        # kept: invoice 1 from sheet a only. Removed: repeat in sheet b, cancellation,
        # zero quantity, zero price, postage.
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean["revenue"].iloc[0], 6.0)

    def test_weekly_fills_empty_weeks(self):
        clean = pd.DataFrame({
            "Invoice": ["1", "2"], "StockCode": ["10001", "10001"], "Quantity": [1, 1],
            "InvoiceDate": pd.to_datetime(["2010-01-04", "2010-01-24"]),   # Monday, then Sunday 3 weeks on
            "Price": [1.0, 1.0], "revenue": [1.0, 1.0],
        })
        weekly = pr.to_weekly(clean)
        self.assertEqual(len(weekly), 3)
        self.assertEqual(weekly["units"].tolist(), [1, 0, 1])


if __name__ == "__main__":
    unittest.main()
