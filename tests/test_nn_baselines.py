"""#114：基準線的技巧分數，以及階數只在驗證期挑。"""

import numpy as np

from trend_detective.nn.baselines import evaluate_series, score_forecast
from trend_detective.nn.data import HOLDOUT_FIRST


def _months(start: str, count: int) -> np.ndarray:
    first = np.datetime64(start, "M")
    return first + np.arange(count).astype("timedelta64[M]")


def test_perfect_forecast_has_skill_one_and_climatology_has_skill_zero() -> None:
    actual = np.asarray([1.0, -1.0, 2.0])
    perfect = score_forecast(actual, actual, scale=1.0)
    climatology = score_forecast(actual, np.zeros(3), scale=1.0)
    assert perfect.skill == 1.0
    assert climatology.skill == 0.0


def test_autoregression_order_is_chosen_without_the_holdout() -> None:
    months = _months("2020-01", 84)
    values = np.sin(np.arange(months.shape[0]) * 2 * np.pi / 12) + 0.1
    fill = np.asarray([""] * months.shape[0])
    flags = np.asarray(["ok"] * months.shape[0])
    evaluated = evaluate_series(months, values, fill, flags, horizons=(1,))
    assert 1 in evaluated["validation"]
    assert 1 in evaluated["holdout"]
    assert np.all(months[months >= HOLDOUT_FIRST] >= HOLDOUT_FIRST)
