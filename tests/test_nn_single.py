"""#115：模型只在訓練期更新，留出期只評估。"""

import numpy as np

from trend_detective.nn.single_series import (
    beats_baseline,
    build_rows,
    fit_network,
    predict,
)


def _months(start: str, count: int) -> np.ndarray:
    first = np.datetime64(start, "M")
    return first + np.arange(count).astype("timedelta64[M]")


def test_holdout_rows_are_not_labeled_train() -> None:
    months = _months("2022-01", 60)
    anomaly = np.sin(np.arange(months.shape[0]) / 3)
    observed = np.ones(months.shape[0], dtype=bool)
    rows = build_rows(months, anomaly, observed, length=6, horizon=1, scale=1.0)
    assert "train" in rows.split
    assert rows.x.shape[1] == 9


def test_fit_ignores_holdout_when_measuring_validation_loss() -> None:
    months = _months("2022-01", 60)
    anomaly = np.linspace(-1.0, 1.0, months.shape[0])
    observed = np.ones(months.shape[0], dtype=bool)
    rows = build_rows(months, anomaly, observed, length=6, horizon=1, scale=1.0)
    names = rows.split
    features = rows.x
    target = rows.y
    changed = target.copy()
    changed[names == "holdout"] = 50.0
    first = fit_network(
        features,
        target,
        names == "train",
        names == "val",
        kind="mlp",
        seed=0,
        quantiles=False,
        learning_rate=1e-2,
    )
    second = fit_network(
        features,
        changed,
        names == "train",
        names == "val",
        kind="mlp",
        seed=0,
        quantiles=False,
        learning_rate=1e-2,
    )
    original = predict(first, features[names == "train"])
    altered = predict(second, features[names == "train"])
    assert np.allclose(original, altered)


def test_beats_baseline_requires_a_gap_larger_than_the_seed_spread() -> None:
    assert beats_baseline(0.4, 0.05, 0.2)
    assert not beats_baseline(0.22, 0.05, 0.2)
    assert not beats_baseline(None, 0.05, 0.2)
