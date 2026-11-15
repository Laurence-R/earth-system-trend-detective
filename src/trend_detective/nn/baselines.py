"""五種預報基準（#114）。目標是訓練期重算的月距平，不是匯出檔裡的 anomaly。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trend_detective.nn.data import (
    HOLDOUT_FIRST,
    TRAIN_LAST,
    VAL_FIRST,
    VAL_LAST,
    ObservationTable,
    SeriesFilter,
    SeriesTable,
    filter_series,
    real_observation,
    training_anomaly,
)

LAYER_VARIABLES = (
    "imerg_precip",
    "gldas_tair",
    "gldas_soilmoist",
    "gldas_evap",
    "airs_tsurf",
    "airs_h2ovap",
    "merra2_t2m",
    "omi_no2",
    "omi_o3",
    "omi_aerosol",
    "modis_ndvi",
    "modis_lst",
)
TAIWAN_POINTS = ("taichung", "taipei", "kaohsiung", "hualien", "yushan")
HORIZONS = (1, 3, 6, 12)
BASELINES = ("climatology", "trend", "persistence", "seasonal", "autoregression")
AR_ORDERS = range(1, 13)


def world_city_ids(path: Path) -> tuple[str, ...]:
    locations = json.loads(path.read_text(encoding="utf-8"))
    return tuple(item["id"] for item in locations if item.get("group") == "world_city")


@dataclass(frozen=True)
class Score:
    n: int
    mae: float
    rmse: float
    mae_standardized: float
    rmse_standardized: float
    skill: float | None


def score_forecast(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    scale: float,
) -> Score:
    error = actual - predicted
    mse = float(np.mean(error**2))
    climatology_mse = float(np.mean(actual**2))
    skill = None if climatology_mse == 0.0 else 1.0 - mse / climatology_mse
    std = 1.0 if not np.isfinite(scale) or scale == 0.0 else scale
    scaled = error / std
    return Score(
        n=int(actual.shape[0]),
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(mse)),
        mae_standardized=float(np.mean(np.abs(scaled))),
        rmse_standardized=float(np.sqrt(np.mean(scaled**2))),
        skill=skill,
    )


def _lookup(by_month: dict[np.datetime64, float], month: np.datetime64) -> float:
    return by_month.get(month, float("nan"))


def _fit_trend(
    months: NDArray[np.datetime64], values: NDArray[np.float64]
) -> tuple[float, float]:
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return 0.0, 0.0
    clock = months[finite].astype(np.int64).astype(np.float64)
    design = np.column_stack([np.ones(clock.shape[0]), clock])
    coef, *_ = np.linalg.lstsq(design, values[finite], rcond=None)
    return float(coef[0]), float(coef[1])


def _fit_ar(
    by_month: dict[np.datetime64, float],
    targets: list[np.datetime64],
    horizon: int,
    order: int,
) -> NDArray[np.float64] | None:
    rows: list[list[float]] = []
    targets_y: list[float] = []
    for target in targets:
        features = [
            _lookup(by_month, target - np.timedelta64(horizon + lag, "M"))
            for lag in range(order)
        ]
        actual = _lookup(by_month, target)
        if np.isfinite(features).all() and np.isfinite(actual):
            rows.append(features)
            targets_y.append(actual)
    if len(rows) < order + 1:
        return None
    design = np.column_stack([np.ones(len(rows)), np.asarray(rows)])
    coef, *_ = np.linalg.lstsq(design, np.asarray(targets_y), rcond=None)
    return coef


def _predict_ar(
    coef: NDArray[np.float64],
    by_month: dict[np.datetime64, float],
    target: np.datetime64,
    horizon: int,
) -> float:
    features = [
        _lookup(by_month, target - np.timedelta64(horizon + lag, "M"))
        for lag in range(coef.shape[0] - 1)
    ]
    if not np.isfinite(features).all():
        return float("nan")
    return float(coef[0] + np.dot(coef[1:], features))


def evaluate_series(
    months: NDArray[np.datetime64],
    values: NDArray[np.float64],
    fill_method: NDArray[np.str_],
    qc_flag: NDArray[np.str_],
    *,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, dict[int, dict[str, Score]]]:
    """回傳 split -> horizon -> baseline -> 分數。自迴歸的階數只在驗證期挑。"""
    anomaly = training_anomaly(values, months)
    observed = real_observation(fill_method, qc_flag, values)
    by_month = {
        months[index]: float(anomaly[index])
        for index in range(months.shape[0])
        if np.isfinite(anomaly[index])
    }
    train_months = months[(months <= TRAIN_LAST) & np.isfinite(anomaly)]
    train_values = anomaly[(months <= TRAIN_LAST) & np.isfinite(anomaly)]
    intercept, slope = _fit_trend(train_months, train_values)
    scale = float(np.std(train_values)) if train_values.size else 1.0
    chosen_order: dict[int, int] = {}
    chosen_coef: dict[int, NDArray[np.float64]] = {}
    train_targets = [
        months[index]
        for index in range(months.shape[0])
        if months[index] <= TRAIN_LAST
        and observed[index]
        and np.isfinite(anomaly[index])
    ]
    validation_targets = [
        months[index]
        for index in range(months.shape[0])
        if VAL_FIRST <= months[index] <= VAL_LAST
        and observed[index]
        and np.isfinite(anomaly[index])
    ]
    for horizon in horizons:
        best_order = 1
        best_mae = float("inf")
        best_coef: NDArray[np.float64] | None = None
        for order in AR_ORDERS:
            coef = _fit_ar(by_month, train_targets, horizon, order)
            if coef is None:
                continue
            errors: list[float] = []
            for target in validation_targets:
                predicted = _predict_ar(coef, by_month, target, horizon)
                actual = _lookup(by_month, target)
                if np.isfinite(predicted) and np.isfinite(actual):
                    errors.append(abs(actual - predicted))
            if not errors:
                continue
            mae = float(np.mean(errors))
            if mae < best_mae:
                best_mae = mae
                best_order = order
                best_coef = coef
        if best_coef is None:
            best_coef = np.asarray([0.0])
        chosen_order[horizon] = best_order
        chosen_coef[horizon] = best_coef
    results: dict[str, dict[int, dict[str, Score]]] = {}
    for split, selected in (
        ("validation", (months >= VAL_FIRST) & (months <= VAL_LAST) & observed),
        ("holdout", (months >= HOLDOUT_FIRST) & observed),
    ):
        results[split] = {}
        for horizon in horizons:
            actuals: dict[str, list[float]] = {name: [] for name in BASELINES}
            predictions: dict[str, list[float]] = {name: [] for name in BASELINES}
            for index in np.flatnonzero(selected):
                target = months[index]
                actual = float(anomaly[index])
                if not np.isfinite(actual):
                    continue
                origin = target - np.timedelta64(horizon, "M")
                persisted = _lookup(by_month, origin)
                seasonal = _lookup(by_month, target - np.timedelta64(12, "M"))
                clock = float(target.astype(np.int64))
                trend = intercept + slope * clock
                autoregression = _predict_ar(
                    chosen_coef[horizon], by_month, target, horizon
                )
                candidates = {
                    "climatology": 0.0,
                    "trend": trend,
                    "persistence": persisted,
                    "seasonal": seasonal,
                    "autoregression": autoregression,
                }
                for name, predicted in candidates.items():
                    if np.isfinite(predicted):
                        actuals[name].append(actual)
                        predictions[name].append(predicted)
            results[split][horizon] = {}
            for name in BASELINES:
                if not actuals[name]:
                    continue
                results[split][horizon][name] = score_forecast(
                    np.asarray(actuals[name]),
                    np.asarray(predictions[name]),
                    scale,
                )
            results[split][horizon]["autoregression_order"] = chosen_order[horizon]  # type: ignore[assignment]
    return results


def _median_iqr(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "q25": None, "q75": None, "n_series": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "n_series": len(values),
    }


def aggregate(
    records: list[dict[str, object]],
) -> dict[str, object]:
    """records 每筆是一條序列在一個 split、一個 h、一個基準的分數。"""

    def collect(
        split: str, horizon: int | None, baseline: str, key: str, **equals: object
    ) -> list[float]:
        chosen: list[float] = []
        for record in records:
            if record["split"] != split or record["baseline"] != baseline:
                continue
            if horizon is not None and record["horizon"] != horizon:
                continue
            if any(record.get(field) != value for field, value in equals.items()):
                continue
            number = record.get(key)
            if isinstance(number, float) and np.isfinite(number):
                chosen.append(number)
        return chosen

    tables: dict[str, object] = {}
    for split in ("validation", "holdout"):
        by_horizon = []
        for horizon in HORIZONS:
            row: dict[str, object] = {"horizon": horizon}
            for baseline in BASELINES:
                row[baseline] = {
                    "mae": _median_iqr(collect(split, horizon, baseline, "mae")),
                    "skill": _median_iqr(collect(split, horizon, baseline, "skill")),
                }
            by_horizon.append(row)
        by_variable = []
        variables = sorted({str(record["variable_id"]) for record in records})
        for variable in variables:
            entry: dict[str, object] = {"variable_id": variable, "horizons": []}
            horizons = entry["horizons"]
            assert isinstance(horizons, list)
            for horizon in HORIZONS:
                row = {"horizon": horizon}
                for baseline in BASELINES:
                    row[baseline] = _median_iqr(
                        collect(
                            split,
                            horizon,
                            baseline,
                            "skill",
                            variable_id=variable,
                        )
                    )
                horizons.append(row)
            by_variable.append(entry)
        by_kind = []
        kinds = sorted({str(record["data_kind"]) for record in records})
        for kind in kinds:
            entry = {"data_kind": kind, "horizons": []}
            horizons = entry["horizons"]
            assert isinstance(horizons, list)
            for horizon in HORIZONS:
                row = {"horizon": horizon}
                for baseline in BASELINES:
                    row[baseline] = _median_iqr(
                        collect(split, horizon, baseline, "skill", data_kind=kind)
                    )
                horizons.append(row)
            by_kind.append(entry)
        tables[split] = {
            "by_horizon": by_horizon,
            "by_variable": by_variable,
            "by_data_kind": by_kind,
        }
    return tables


def iter_series_records(
    series: SeriesTable, observations: ObservationTable
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(len(series)):
        series_id = int(series.series_id[index])
        chosen = observations.series_id == series_id
        if not np.any(chosen):
            continue
        evaluated = evaluate_series(
            observations.month[chosen],
            observations.value_clean[chosen],
            observations.fill_method[chosen],
            observations.qc_flag[chosen],
        )
        for split, horizons in evaluated.items():
            for horizon, baselines in horizons.items():
                for name, value in baselines.items():
                    if not isinstance(value, Score):
                        continue
                    records.append(
                        {
                            "series_id": series_id,
                            "variable_id": str(series.variable_id[index]),
                            "location_id": str(series.location_id[index]),
                            "data_kind": str(series.data_kind[index]),
                            "split": split,
                            "horizon": horizon,
                            "baseline": name,
                            "n": value.n,
                            "mae": value.mae,
                            "rmse": value.rmse,
                            "mae_standardized": value.mae_standardized,
                            "rmse_standardized": value.rmse_standardized,
                            "skill": value.skill,
                        }
                    )
    return records


def select_layer_series(series: SeriesTable, cities: tuple[str, ...]) -> SeriesTable:
    locations = TAIWAN_POINTS + cities
    mask = filter_series(
        series,
        SeriesFilter(
            variable_ids=LAYER_VARIABLES,
            location_ids=locations,
        ),
    )
    point = series.aggregation == "point"
    return series.select(mask & point)


def run(
    series: SeriesTable, observations: ObservationTable, cities: tuple[str, ...]
) -> dict[str, object]:
    selected = select_layer_series(series, cities)
    ids = {int(item) for item in selected.series_id}
    observed = np.fromiter(
        (int(item) in ids for item in observations.series_id),
        dtype=bool,
        count=len(observations),
    )
    records = iter_series_records(
        selected,
        ObservationTable(
            series_id=observations.series_id[observed],
            month=observations.month[observed],
            value_raw=observations.value_raw[observed],
            value_clean=observations.value_clean[observed],
            fill_method=observations.fill_method[observed],
            qc_flag=observations.qc_flag[observed],
        ),
    )
    return {
        "exploratory": True,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameters": {
            "horizons": list(HORIZONS),
            "train_last": "2023-12",
            "validation": "2024-01/2025-12",
            "holdout": "2026-01 onward, evaluated once in this file",
            "persistence": "anomaly at the forecast origin, which is the target month minus h",
            "seasonal": "anomaly of the same calendar month one year earlier",
            "autoregression": "order 1-12 chosen by validation MAE, fit on the training period only",
            "target": "value_clean minus the training-period calendar-month mean",
            "variables": list(LAYER_VARIABLES),
            "locations": list(TAIWAN_POINTS + cities),
        },
        "n_series": len(selected),
        "tables": aggregate(records),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from trend_detective.nn.data import load_database

    parser = argparse.ArgumentParser(description="計算五種預報基準")
    parser.add_argument(
        "--cities",
        type=Path,
        default=Path("data/seed/locations.world.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis/nn/baselines.json"),
    )
    args = parser.parse_args(argv)
    series, observations = load_database()
    payload = run(series, observations, world_city_ids(args.cities))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    run_path = args.output.with_suffix(".run.json")
    run_path.write_text(
        json.dumps(
            {
                "command": "uv run python -m trend_detective.nn.baselines",
                "generated_at": payload["generated_at"],
                "exploratory": True,
                "holdout_evaluated_once": True,
                "n_series": payload["n_series"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({payload['n_series']} series)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
