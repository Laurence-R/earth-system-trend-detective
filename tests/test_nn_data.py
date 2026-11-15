"""#113：時間切分不得把留出期間漏進訓練或驗證。"""

from pathlib import Path

import numpy as np

from trend_detective.nn.data import (
    HOLDOUT_FIRST,
    ObservationTable,
    SeriesFilter,
    SeriesTable,
    filter_series,
    fold_masks,
    leave_one_location,
    load_csv,
    make_windows,
    real_observation,
    split_names,
    train_scale,
    walk_forward_cuts,
    write_export,
)


def _months(start: str, count: int) -> np.ndarray:
    first = np.datetime64(start, "M")
    return first + np.arange(count).astype("timedelta64[M]")


def _series() -> SeriesTable:
    return SeriesTable(
        series_id=np.asarray([1, 2], dtype=np.int64),
        variable_id=np.asarray(["imerg_precip", "gldas_tair"]),
        location_id=np.asarray(["taichung", "taipei"]),
        aggregation=np.asarray(["point", "point"]),
        tier=np.asarray(["core", "core"]),
        unit=np.asarray(["mm/hr", "K"]),
        data_kind=np.asarray(["satellite_retrieval", "model"]),
        is_auxiliary=np.asarray([False, True], dtype=np.bool_),
        category_path=np.asarray(["water", "energy"]),
        location_group=np.asarray(["taiwan", "taiwan"]),
        lat=np.asarray([24.15, 25.03]),
        lon=np.asarray([120.68, 121.56]),
        n_usable=np.asarray([200, 10], dtype=np.int64),
        fill_fraction=np.asarray([0.0, 0.4]),
        mining_eligible=np.asarray([True, False], dtype=np.bool_),
        known_issues=np.asarray(["", "gldas_tair_steps"]),
    )


def test_training_windows_stay_before_validation_and_holdout() -> None:
    months = _months("2022-01", 60)
    values = np.arange(months.shape[0], dtype=np.float64)
    fill = np.asarray([""] * months.shape[0])
    flags = np.asarray(["ok"] * months.shape[0])
    batch = make_windows(1, months, values, fill, flags, length=6, horizon=1)
    train = batch.split == "train"
    assert train.any()
    assert np.all(batch.target_month[train] < np.datetime64("2024-01", "M"))
    assert np.all(batch.input_end[train] < np.datetime64("2024-01", "M"))
    usable = batch.split != "holdout"
    assert np.all(batch.target_month[usable] < HOLDOUT_FIRST)
    assert np.all(batch.input_end[usable] < HOLDOUT_FIRST)


def test_holdout_month_inside_the_window_is_not_validation() -> None:
    months = _months("2025-08", 8)
    values = np.arange(months.shape[0], dtype=np.float64)
    fill = np.asarray([""] * months.shape[0])
    flags = np.asarray(["ok"] * months.shape[0])
    batch = make_windows(1, months, values, fill, flags, length=4, horizon=3)
    leaked = (batch.split == "val") & (batch.target_month >= HOLDOUT_FIRST)
    assert not leaked.any()
    assert np.all(batch.split[batch.target_month >= HOLDOUT_FIRST] == "holdout")


def test_scale_ignores_validation_and_holdout_values() -> None:
    months = _months("2023-01", 48)
    values = np.linspace(0.0, 10.0, months.shape[0])
    original = train_scale(values, months)
    changed = values.copy()
    changed[months >= np.datetime64("2024-01", "M")] = 10_000.0
    assert train_scale(changed, months) == original


def test_filled_or_flagged_months_are_not_real_observations() -> None:
    values = np.asarray([1.0, 2.0, 3.0, np.nan])
    fill = np.asarray(["", "linear", "", ""])
    flags = np.asarray(["ok", "ok", "low_coverage", "ok"])
    assert real_observation(fill, flags, values).tolist() == [True, False, False, False]


def test_filter_drops_auxiliary_and_known_issues() -> None:
    series = _series()
    mask = filter_series(
        series,
        SeriesFilter(
            exclude_auxiliary=True, exclude_known_issues=True, min_n_usable=100
        ),
    )
    assert series.variable_id[mask].tolist() == ["imerg_precip"]


def test_walk_forward_and_leave_one_location_keep_holdout_out() -> None:
    months = _months("2015-01", 144)
    for val_first, val_last in walk_forward_cuts():
        train, validation = fold_masks(months, val_first, val_last)
        assert np.all(months[train] < HOLDOUT_FIRST)
        assert np.all(months[validation] < HOLDOUT_FIRST)
        assert val_last < HOLDOUT_FIRST
    locations = np.asarray(["taichung", "taipei", "taichung"])
    train, test = leave_one_location(locations, "taichung")
    assert locations[test].tolist() == ["taichung", "taichung"]
    assert locations[train].tolist() == ["taipei"]


def test_split_names_cover_the_three_periods() -> None:
    months = np.asarray(
        ["2023-12", "2024-01", "2025-12", "2026-01"], dtype="datetime64[M]"
    )
    assert split_names(months).tolist() == ["train", "val", "val", "holdout"]


def test_csv_round_trip(tmp_path: Path) -> None:
    series = _series().select(np.asarray([True, False], dtype=np.bool_))
    months = _months("2023-11", 3)
    observations = ObservationTable(
        series_id=np.asarray([1, 1, 1], dtype=np.int64),
        month=months,
        value_raw=np.asarray([1.0, 2.0, np.nan]),
        value_clean=np.asarray([1.0, np.nan, 3.0]),
        fill_method=np.asarray(["", "linear", ""]),
        qc_flag=np.asarray(["ok", "ok", "ok"]),
    )
    series_path, obs_path = write_export(series, observations, tmp_path)
    loaded_series, loaded_obs = load_csv(series_path, obs_path)
    assert loaded_series.variable_id.tolist() == ["imerg_precip"]
    assert loaded_obs.fill_method.tolist() == ["", "linear", ""]
    assert np.isnan(loaded_obs.value_clean[1])
