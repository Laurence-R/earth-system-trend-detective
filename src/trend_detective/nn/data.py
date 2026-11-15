"""讀清洗後的序列，並依時間切分。

切分、視窗、標準化不連資料庫。匯出檔的 anomaly 含全期月氣候值，這裡不用它當目標。
"""

from __future__ import annotations

import csv
import gzip
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

TRAIN_LAST = np.datetime64("2023-12", "M")
VAL_FIRST = np.datetime64("2024-01", "M")
VAL_LAST = np.datetime64("2025-12", "M")
HOLDOUT_FIRST = np.datetime64("2026-01", "M")
WALK_STEP_MONTHS = 24
WALK_FIRST = np.datetime64("2016-01", "M")

SERIES_SQL = """
SELECT s.series_id, s.variable_id, s.location_id, s.aggregation, s.tier, s.unit,
       v.data_kind, v.is_auxiliary, COALESCE(v.category_path::text, ''),
       l.location_group, l.lat, l.lon,
       ss.n_usable, ss.fill_fraction, ss.mining_eligible,
       COALESCE(array_to_string(ss.known_issues, ','), '')
FROM meta.series AS s
JOIN meta.variables AS v ON v.id = s.variable_id
JOIN meta.locations AS l ON l.id = s.location_id
JOIN clean.series_summary AS ss ON ss.series_id = s.series_id
ORDER BY s.series_id
"""

OBS_SQL = """
SELECT series_id, month, value_raw, value_clean, fill_method, qc_flag
FROM clean.observations
ORDER BY series_id, month
"""


def to_month(value: date | str | np.datetime64) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[M]")
    return np.datetime64(str(value)[:7], "M")


def split_names(months: NDArray[np.datetime64]) -> NDArray[np.str_]:
    """訓練到 2023-12，驗證 2024–2025，2026-01 起為留出。"""
    names = np.empty(months.shape, dtype=object)
    names[months <= TRAIN_LAST] = "train"
    validation = (months >= VAL_FIRST) & (months <= VAL_LAST)
    names[validation] = "val"
    names[months >= HOLDOUT_FIRST] = "holdout"
    return names.astype(str)


@dataclass(frozen=True)
class SeriesTable:
    series_id: NDArray[np.int64]
    variable_id: NDArray[np.str_]
    location_id: NDArray[np.str_]
    aggregation: NDArray[np.str_]
    tier: NDArray[np.str_]
    unit: NDArray[np.str_]
    data_kind: NDArray[np.str_]
    is_auxiliary: NDArray[np.bool_]
    category_path: NDArray[np.str_]
    location_group: NDArray[np.str_]
    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    n_usable: NDArray[np.int64]
    fill_fraction: NDArray[np.float64]
    mining_eligible: NDArray[np.bool_]
    known_issues: NDArray[np.str_]

    def __len__(self) -> int:
        return int(self.series_id.shape[0])

    def select(self, mask: NDArray[np.bool_]) -> SeriesTable:
        picked = {item.name: getattr(self, item.name)[mask] for item in fields(self)}
        return SeriesTable(**picked)


@dataclass(frozen=True)
class ObservationTable:
    series_id: NDArray[np.int64]
    month: NDArray[np.datetime64]
    value_raw: NDArray[np.float64]
    value_clean: NDArray[np.float64]
    fill_method: NDArray[np.str_]
    qc_flag: NDArray[np.str_]

    def __len__(self) -> int:
        return int(self.series_id.shape[0])


@dataclass(frozen=True)
class SeriesFilter:
    tiers: tuple[str, ...] | None = None
    data_kinds: tuple[str, ...] | None = None
    variable_ids: tuple[str, ...] | None = None
    location_ids: tuple[str, ...] | None = None
    location_groups: tuple[str, ...] | None = None
    exclude_auxiliary: bool = False
    exclude_known_issues: bool = False
    min_n_usable: int | None = None


def filter_series(series: SeriesTable, spec: SeriesFilter) -> NDArray[np.bool_]:
    mask = np.ones(len(series), dtype=bool)
    if spec.tiers is not None:
        mask &= np.isin(series.tier, list(spec.tiers))
    if spec.data_kinds is not None:
        mask &= np.isin(series.data_kind, list(spec.data_kinds))
    if spec.variable_ids is not None:
        mask &= np.isin(series.variable_id, list(spec.variable_ids))
    if spec.location_ids is not None:
        mask &= np.isin(series.location_id, list(spec.location_ids))
    if spec.location_groups is not None:
        mask &= np.isin(series.location_group, list(spec.location_groups))
    if spec.exclude_auxiliary:
        mask &= ~series.is_auxiliary
    if spec.exclude_known_issues:
        mask &= series.known_issues == ""
    if spec.min_n_usable is not None:
        mask &= series.n_usable >= spec.min_n_usable
    return mask


def real_observation(
    fill_method: NDArray[np.str_],
    qc_flag: NDArray[np.str_],
    value: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """補值與非 ok 的月份不算真實觀測。"""
    return (fill_method == "") & (qc_flag == "ok") & np.isfinite(value)


def train_scale(
    values: NDArray[np.float64], months: NDArray[np.datetime64]
) -> tuple[float, float]:
    """平均與標準差只看訓練期有值的 value_clean。"""
    chosen = values[(months <= TRAIN_LAST) & np.isfinite(values)]
    if chosen.size == 0:
        return float("nan"), float("nan")
    mean = float(chosen.mean())
    std = float(chosen.std(ddof=0))
    if std == 0.0:
        std = 1.0
    return mean, std


def calendar_means(
    values: NDArray[np.float64], months: NDArray[np.datetime64]
) -> NDArray[np.float64]:
    """12 個日曆月的平均，只使用訓練期的 value_clean。"""
    means = np.full(12, np.nan)
    train = (months <= TRAIN_LAST) & np.isfinite(values)
    month_numbers = months.astype("datetime64[M]").astype(int) % 12
    for index in range(12):
        chosen = values[train & (month_numbers == index)]
        if chosen.size:
            means[index] = float(chosen.mean())
    return means


def training_anomaly(
    values: NDArray[np.float64], months: NDArray[np.datetime64]
) -> NDArray[np.float64]:
    means = calendar_means(values, months)
    month_numbers = months.astype("datetime64[M]").astype(int) % 12
    anomaly = np.full(values.shape, np.nan)
    finite = np.isfinite(values) & np.isfinite(means[month_numbers])
    anomaly[finite] = values[finite] - means[month_numbers][finite]
    return anomaly


@dataclass(frozen=True)
class WindowBatch:
    series_id: NDArray[np.int64]
    input_end: NDArray[np.datetime64]
    target_month: NDArray[np.datetime64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    observed: NDArray[np.bool_]
    split: NDArray[np.str_]


def make_windows(
    series_id: int,
    months: NDArray[np.datetime64],
    values: NDArray[np.float64],
    fill_method: NDArray[np.str_],
    qc_flag: NDArray[np.str_],
    *,
    length: int,
    horizon: int,
) -> WindowBatch:
    """輸入是連續 length 個月；目標是輸入結束後再過 horizon 個月。

    標準化只用這條序列的訓練期。留出月份出現在輸入或目標時，樣本不是訓練或驗證。
    """
    order = np.argsort(months, kind="stable")
    months = months[order]
    values = values[order]
    fill_method = fill_method[order]
    qc_flag = qc_flag[order]
    mean, std = train_scale(values, months)
    by_month = {months[index]: index for index in range(int(months.shape[0]))}
    ids: list[int] = []
    ends: list[np.datetime64] = []
    targets: list[np.datetime64] = []
    rows: list[NDArray[np.float64]] = []
    ys: list[float] = []
    observed_rows: list[bool] = []
    splits: list[str] = []
    for end in months:
        start = end - np.timedelta64(length - 1, "M")
        window = [start + np.timedelta64(offset, "M") for offset in range(length)]
        if any(item not in by_month for item in window):
            continue
        target = end + np.timedelta64(horizon, "M")
        features = np.empty(length, dtype=np.float64)
        for feature_index, item in enumerate(window):
            raw = float(values[by_month[item]])
            features[feature_index] = (raw - mean) / std if np.isfinite(raw) else np.nan
        target_index = by_month.get(target)
        if target_index is None:
            y_value = float("nan")
            is_real = False
        else:
            y_value = float(values[target_index])
            is_real = (
                fill_method[target_index] == ""
                and qc_flag[target_index] == "ok"
                and bool(np.isfinite(y_value))
            )
        ids.append(series_id)
        ends.append(end)
        targets.append(target)
        rows.append(features)
        ys.append(y_value)
        observed_rows.append(is_real)
        splits.append(_sample_split([*window, target]))
    length_columns = length
    if not ids:
        empty_month = np.empty(0, dtype="datetime64[M]")
        return WindowBatch(
            series_id=np.empty(0, dtype=np.int64),
            input_end=empty_month,
            target_month=empty_month,
            x=np.empty((0, length_columns)),
            y=np.empty(0),
            observed=np.empty(0, dtype=bool),
            split=np.empty(0, dtype=str),
        )
    return WindowBatch(
        series_id=np.asarray(ids, dtype=np.int64),
        input_end=np.asarray(ends, dtype="datetime64[M]"),
        target_month=np.asarray(targets, dtype="datetime64[M]"),
        x=np.vstack(rows),
        y=np.asarray(ys, dtype=np.float64),
        observed=np.asarray(observed_rows, dtype=bool),
        split=np.asarray(splits, dtype=str),
    )


def _sample_split(months: Sequence[np.datetime64]) -> str:
    names = split_names(np.asarray(list(months), dtype="datetime64[M]"))
    if np.any(names == "holdout"):
        return "holdout"
    if np.any(names == "val"):
        return "val"
    return "train"


def walk_forward_cuts(
    *,
    first: np.datetime64 = WALK_FIRST,
    step: int = WALK_STEP_MONTHS,
) -> list[tuple[np.datetime64, np.datetime64]]:
    """每折的驗證段往後推 step 個月，且不進入留出期間。"""
    cuts: list[tuple[np.datetime64, np.datetime64]] = []
    val_first = first
    while val_first <= VAL_LAST:
        val_last = min(val_first + np.timedelta64(step - 1, "M"), VAL_LAST)
        cuts.append((val_first, val_last))
        val_first = val_first + np.timedelta64(step, "M")
    return cuts


def fold_masks(
    months: NDArray[np.datetime64],
    val_first: np.datetime64,
    val_last: np.datetime64,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    train = months < val_first
    validation = (months >= val_first) & (months <= val_last)
    return train, validation


def leave_one_location(
    location_ids: NDArray[np.str_], held_out: str
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    test = location_ids == held_out
    return ~test, test


def summarize(
    series: SeriesTable, observations: ObservationTable
) -> list[dict[str, object]]:
    real = real_observation(
        observations.fill_method, observations.qc_flag, observations.value_clean
    )
    groups: dict[tuple[str, str], list[int]] = {}
    for series_id, tier, kind in zip(
        series.series_id, series.tier, series.data_kind, strict=True
    ):
        groups.setdefault((str(tier), str(kind)), []).append(int(series_id))
    rows: list[dict[str, object]] = []
    membership = {
        int(series_id): (str(tier), str(kind))
        for series_id, tier, kind in zip(
            series.series_id, series.tier, series.data_kind, strict=True
        )
    }
    counts: dict[tuple[str, str], list[int]] = {key: [0, 0] for key in groups}
    for series_id, is_real in zip(observations.series_id, real, strict=True):
        key = membership.get(int(series_id))
        if key is None:
            continue
        counts[key][0] += 1
        if not bool(is_real):
            counts[key][1] += 1
    for key in sorted(groups):
        samples, masked = counts[key]
        rows.append(
            {
                "tier": key[0],
                "data_kind": key[1],
                "n_series": len(groups[key]),
                "n_samples": samples,
                "masked_fraction": (masked / samples) if samples else 0.0,
            }
        )
    return rows


def load_csv(
    series_path: Path, observations_path: Path
) -> tuple[SeriesTable, ObservationTable]:
    with series_path.open(encoding="utf-8", newline="") as handle:
        series_rows = list(csv.DictReader(handle))
    opener = gzip.open if observations_path.name.endswith(".gz") else open
    with opener(observations_path, "rt", encoding="utf-8", newline="") as handle:
        obs_rows = list(csv.DictReader(handle))
    series = SeriesTable(
        series_id=np.asarray(
            [int(row["series_id"]) for row in series_rows], dtype=np.int64
        ),
        variable_id=np.asarray([row["variable_id"] for row in series_rows]),
        location_id=np.asarray([row["location_id"] for row in series_rows]),
        aggregation=np.asarray([row["aggregation"] for row in series_rows]),
        tier=np.asarray([row["tier"] for row in series_rows]),
        unit=np.asarray([row.get("unit", "") for row in series_rows]),
        data_kind=np.asarray([row["data_kind"] for row in series_rows]),
        is_auxiliary=np.asarray(
            [row["is_auxiliary"].lower() == "true" for row in series_rows],
            dtype=np.bool_,
        ),
        category_path=np.asarray([row.get("category_path", "") for row in series_rows]),
        location_group=np.asarray(
            [row.get("location_group", "") for row in series_rows]
        ),
        lat=np.asarray([_float_or_nan(row.get("lat", "")) for row in series_rows]),
        lon=np.asarray([_float_or_nan(row.get("lon", "")) for row in series_rows]),
        n_usable=np.asarray(
            [int(row["n_usable"]) for row in series_rows], dtype=np.int64
        ),
        fill_fraction=np.asarray([float(row["fill_fraction"]) for row in series_rows]),
        mining_eligible=np.asarray(
            [
                row.get("mining_eligible", "true").lower() == "true"
                for row in series_rows
            ],
            dtype=np.bool_,
        ),
        known_issues=np.asarray([row.get("known_issues", "") for row in series_rows]),
    )
    observations = ObservationTable(
        series_id=np.asarray(
            [int(row["series_id"]) for row in obs_rows], dtype=np.int64
        ),
        month=np.asarray([to_month(row["month"]) for row in obs_rows]),
        value_raw=np.asarray([_float_or_nan(row["value_raw"]) for row in obs_rows]),
        value_clean=np.asarray([_float_or_nan(row["value_clean"]) for row in obs_rows]),
        fill_method=np.asarray([row["fill_method"] for row in obs_rows]),
        qc_flag=np.asarray([row["qc_flag"] for row in obs_rows]),
    )
    return series, observations


def _float_or_nan(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def load_database(env_file: Path | None = None) -> tuple[SeriesTable, ObservationTable]:
    """連得上 `.env` 就用它；否則經本機容器查，不印出密碼。"""
    from trend_detective.pg import CONTAINER, ENV_FILE, connect, pg_settings

    chosen = ENV_FILE if env_file is None else env_file
    if "PGPASSWORD" in pg_settings(chosen):
        with connect(chosen) as connection:
            series_rows = connection.execute(SERIES_SQL).fetchall()
            obs_rows = connection.execute(OBS_SQL).fetchall()
        return _series_from_rows(series_rows), _observations_from_rows(obs_rows)
    series_rows = _copy_via_docker(SERIES_SQL, CONTAINER)
    obs_rows = _copy_via_docker(OBS_SQL, CONTAINER)
    return _series_from_rows(_typed_series(series_rows)), _observations_from_rows(
        _typed_observations(obs_rows)
    )


def _copy_via_docker(query: str, container: str) -> list[list[str]]:
    import subprocess

    statement = f"COPY ({query.rstrip().rstrip(';')}) TO STDOUT WITH (FORMAT csv)"
    completed = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "psql",
            "-U",
            "trend",
            "-d",
            "trend_detective",
            "-q",
            "-c",
            statement,
        ],
        check=True,
        capture_output=True,
    )
    text = completed.stdout.decode("utf-8")
    return list(csv.reader(text.splitlines()))


def _typed_series(rows: list[list[str]]) -> list[tuple[object, ...]]:
    typed: list[tuple[object, ...]] = []
    for row in rows:
        typed.append(
            (
                int(row[0]),
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7] == "t",
                row[8],
                row[9],
                None if row[10] == "" else float(row[10]),
                None if row[11] == "" else float(row[11]),
                int(row[12] or 0),
                float("nan") if row[13] == "" else float(row[13]),
                row[14] == "t",
                row[15],
            )
        )
    return typed


def _typed_observations(rows: list[list[str]]) -> list[tuple[object, ...]]:
    typed: list[tuple[object, ...]] = []
    for row in rows:
        typed.append(
            (
                int(row[0]),
                row[1],
                None if row[2] == "" else float(row[2]),
                None if row[3] == "" else float(row[3]),
                None if row[4] == "" else row[4],
                row[5],
            )
        )
    return typed


def _blank(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _series_from_rows(rows: Sequence[tuple[object, ...]]) -> SeriesTable:
    if not rows:
        raise ValueError("序列查詢沒有列")
    columns: list[tuple[object, ...]] = [
        tuple(column) for column in zip(*rows, strict=True)
    ]

    def text(column: tuple[object, ...]) -> NDArray[np.str_]:
        return np.asarray(["" if item is None else str(item) for item in column])

    return SeriesTable(
        series_id=np.asarray(columns[0], dtype=np.int64),
        variable_id=text(columns[1]),
        location_id=text(columns[2]),
        aggregation=text(columns[3]),
        tier=text(columns[4]),
        unit=text(columns[5]),
        data_kind=text(columns[6]),
        is_auxiliary=np.asarray([bool(item) for item in columns[7]], dtype=np.bool_),
        category_path=text(columns[8]),
        location_group=text(columns[9]),
        lat=np.asarray(
            [np.nan if item is None else float(item) for item in columns[10]]  # type: ignore[arg-type]
        ),
        lon=np.asarray(
            [np.nan if item is None else float(item) for item in columns[11]]  # type: ignore[arg-type]
        ),
        n_usable=np.asarray(columns[12], dtype=np.int64),
        fill_fraction=np.asarray(columns[13], dtype=np.float64),
        mining_eligible=np.asarray(
            [bool(item) for item in columns[14]], dtype=np.bool_
        ),
        known_issues=text(columns[15]),
    )


def _observations_from_rows(rows: Iterable[tuple[object, ...]]) -> ObservationTable:
    series_ids: list[int] = []
    months: list[np.datetime64] = []
    raws: list[float] = []
    cleans: list[float] = []
    fills: list[str] = []
    flags: list[str] = []
    for series_id, month, value_raw, value_clean, fill_method, qc_flag in rows:
        series_ids.append(int(series_id))  # type: ignore[arg-type]
        months.append(to_month(month))  # type: ignore[arg-type]
        raws.append(np.nan if value_raw is None else float(value_raw))  # type: ignore[arg-type]
        cleans.append(np.nan if value_clean is None else float(value_clean))  # type: ignore[arg-type]
        fills.append(_blank(fill_method))
        flags.append(str(qc_flag))
    return ObservationTable(
        series_id=np.asarray(series_ids, dtype=np.int64),
        month=np.asarray(months, dtype="datetime64[M]"),
        value_raw=np.asarray(raws, dtype=np.float64),
        value_clean=np.asarray(cleans, dtype=np.float64),
        fill_method=np.asarray(fills),
        qc_flag=np.asarray(flags),
    )


def write_export(
    series: SeriesTable, observations: ObservationTable, directory: Path
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    series_path = directory / "series.csv"
    obs_path = directory / "observations.csv.gz"
    with series_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series_id",
                "variable_id",
                "location_id",
                "aggregation",
                "tier",
                "unit",
                "data_kind",
                "is_auxiliary",
                "category_path",
                "location_group",
                "lat",
                "lon",
                "n_usable",
                "fill_fraction",
                "mining_eligible",
                "known_issues",
            ]
        )
        for index in range(len(series)):
            writer.writerow(
                [
                    int(series.series_id[index]),
                    series.variable_id[index],
                    series.location_id[index],
                    series.aggregation[index],
                    series.tier[index],
                    series.unit[index],
                    series.data_kind[index],
                    "true" if series.is_auxiliary[index] else "false",
                    series.category_path[index],
                    series.location_group[index],
                    "" if np.isnan(series.lat[index]) else float(series.lat[index]),
                    "" if np.isnan(series.lon[index]) else float(series.lon[index]),
                    int(series.n_usable[index]),
                    float(series.fill_fraction[index]),
                    "true" if series.mining_eligible[index] else "false",
                    series.known_issues[index],
                ]
            )
    with gzip.open(obs_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["series_id", "month", "value_raw", "value_clean", "fill_method", "qc_flag"]
        )
        for index in range(len(observations)):
            month = str(observations.month[index])
            writer.writerow(
                [
                    int(observations.series_id[index]),
                    f"{month}-01",
                    _csv_float(float(observations.value_raw[index])),
                    _csv_float(float(observations.value_clean[index])),
                    observations.fill_method[index],
                    observations.qc_flag[index],
                ]
            )
    return series_path, obs_path


def _csv_float(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return repr(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_summary(rows: Sequence[dict[str, object]]) -> str:
    lines = ["tier\tdata_kind\tn_series\tn_samples\tmasked_fraction"]
    for row in rows:
        fraction = row["masked_fraction"]
        if not isinstance(fraction, float):
            raise TypeError(fraction)
        lines.append(
            f"{row['tier']}\t{row['data_kind']}\t{row['n_series']}\t"
            f"{row['n_samples']}\t{fraction:.6f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from trend_detective.pg import ENV_FILE

    parser = argparse.ArgumentParser(
        description="讀清洗表、寫 ml-export，並印出切分摘要"
    )
    parser.add_argument("--export-dir", type=Path, default=Path("data/ml-export"))
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    args = parser.parse_args(argv)
    series, observations = load_database(args.env_file)
    series_path, obs_path = write_export(series, observations, args.export_dir)
    print(format_summary(summarize(series, observations)))
    print(f"sha256 {series_path} {sha256(series_path)}")
    print(f"sha256 {obs_path} {sha256(obs_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
