"""單一序列的 MLP 與 GRU 預報（#115）。和 #114 用同一批點位、同一種距平。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from trend_detective.nn.baselines import (
    HORIZONS,
    LAYER_VARIABLES,
    TAIWAN_POINTS,
    Score,
    evaluate_series,
    score_forecast,
    world_city_ids,
)
from trend_detective.nn.data import (
    HOLDOUT_FIRST,
    TRAIN_LAST,
    VAL_LAST,
    SeriesTable,
    real_observation,
    training_anomaly,
)

LENGTHS = (12, 24)
LEARNING_RATES = (1e-2, 1e-3)
SEEDS = (0, 1, 2, 3, 4)
CHECK_SERIES = (
    ("imerg_precip", "taichung"),
    ("merra2_t2m", "london"),
    ("omi_no2", "taipei"),
    ("gldas_tair", "yushan"),
    ("modis_ndvi", "kaohsiung"),
)
HIDDEN = 32
GRU_HIDDEN = 16
EPOCHS = 40
PATIENCE = 6


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Rows:
    def __init__(
        self,
        features: NDArray[np.float64],
        target: NDArray[np.float64],
        observed: NDArray[np.bool_],
        split: NDArray[np.str_],
        month: NDArray[np.str_],
    ) -> None:
        self.x = features
        self.y = target
        self.observed = observed
        self.split = split
        self.month = month


def build_rows(
    months: NDArray[np.datetime64],
    anomaly: NDArray[np.float64],
    observed: NDArray[np.bool_],
    *,
    length: int,
    horizon: int,
    scale: float,
) -> Rows:
    """一列是一個目標月。輸入不含目標月之後的值。"""
    by_month = {
        months[index]: float(anomaly[index]) for index in range(months.shape[0])
    }
    features: list[list[float]] = []
    targets: list[float] = []
    flags: list[bool] = []
    names: list[str] = []
    labels: list[str] = []
    for index, target in enumerate(months):
        origin = target - np.timedelta64(horizon, "M")
        window = [
            origin - np.timedelta64(length - 1 - offset, "M")
            for offset in range(length)
        ]
        values = [by_month.get(item, float("nan")) for item in window]
        if not np.isfinite(values).all():
            continue
        actual = float(anomaly[index])
        if target <= TRAIN_LAST:
            split = "train"
        elif target <= VAL_LAST:
            split = "val"
        elif target >= HOLDOUT_FIRST:
            split = "holdout"
        else:
            continue
        angle = 2 * np.pi * (int(target.astype(int)) % 12) / 12
        scaled = [item / scale for item in values]
        features.append(
            [*scaled, float(np.sin(angle)), float(np.cos(angle)), horizon / 12]
        )
        targets.append(actual)
        flags.append(bool(observed[index]) and np.isfinite(actual))
        names.append(split)
        labels.append(str(target)[:7])
    width = length + 3
    if not features:
        return Rows(
            np.empty((0, width)),
            np.empty(0),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=str),
            np.empty(0, dtype=str),
        )
    return Rows(
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(flags, dtype=bool),
        np.asarray(names, dtype=str),
        np.asarray(labels, dtype=str),
    )


class _MLP(nn.Module):
    def __init__(self, width: int, outputs: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN, outputs),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class _GRU(nn.Module):
    def __init__(self, width: int, outputs: int) -> None:
        super().__init__()
        self.length = width - 3
        self.gru = nn.GRU(1, GRU_HIDDEN, batch_first=True)
        self.head = nn.Linear(GRU_HIDDEN + 3, outputs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        series = features[:, : self.length].unsqueeze(-1)
        _output, hidden = self.gru(series)
        extra = features[:, self.length :]
        return self.head(torch.cat([hidden[-1], extra], dim=1))


def _pinball(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=prediction.device)
    error = target.unsqueeze(1) - prediction
    return torch.mean(torch.maximum(quantiles * error, (quantiles - 1) * error))


def fit_network(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    train: NDArray[np.bool_],
    validation: NDArray[np.bool_],
    *,
    kind: str,
    seed: int,
    quantiles: bool,
    learning_rate: float,
) -> nn.Module:
    """只用 train 列更新權重；validation 只決定提早停止。"""
    torch.manual_seed(seed)
    device = _device()
    outputs = 3 if quantiles else 1
    model = (
        _GRU(features.shape[1], outputs)
        if kind == "gru"
        else _MLP(features.shape[1], outputs)
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    x_train = torch.tensor(features[train], dtype=torch.float32, device=device)
    y_train = torch.tensor(target[train], dtype=torch.float32, device=device)
    x_val = torch.tensor(features[validation], dtype=torch.float32, device=device)
    y_val = torch.tensor(target[validation], dtype=torch.float32, device=device)
    best_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    best_loss = float("inf")
    waited = 0
    for _epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        prediction = model(x_train)
        loss = (
            _pinball(prediction, y_train)
            if quantiles
            else torch.mean((prediction.squeeze(1) - y_train) ** 2)
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            if len(y_val) == 0:
                current = float(loss.detach())
            else:
                held = model(x_val)
                current = float(
                    _pinball(held, y_val)
                    if quantiles
                    else torch.mean((held.squeeze(1) - y_val) ** 2)
                )
        if current < best_loss:
            best_loss = current
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            waited = 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model: nn.Module, features: NDArray[np.float64]) -> NDArray[np.float64]:
    device = _device()
    with torch.no_grad():
        output = model(torch.tensor(features, dtype=torch.float32, device=device))
    return output.detach().cpu().numpy()


def coverage(
    lower: NDArray[np.float64], upper: NDArray[np.float64], actual: NDArray[np.float64]
) -> float:
    if actual.size == 0:
        return float("nan")
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


def _score_split(
    prediction: NDArray[np.float64], rows: Rows, split: str, scale: float
) -> Score | None:
    chosen = (rows.split == split) & rows.observed
    if not np.any(chosen):
        return None
    return score_forecast(rows.y[chosen], prediction[chosen], scale)


def series_forecast(
    months: NDArray[np.datetime64],
    values: NDArray[np.float64],
    fill_method: NDArray[np.str_],
    qc_flag: NDArray[np.str_],
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, object]:
    anomaly = training_anomaly(values, months)
    observed = real_observation(fill_method, qc_flag, values)
    train_values = anomaly[(months <= TRAIN_LAST) & np.isfinite(anomaly)]
    scale = float(np.std(train_values)) if train_values.size else 1.0
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    baselines = evaluate_series(months, values, fill_method, qc_flag)
    length, learning_rate = _choose_settings(months, anomaly, observed, scale)
    models: dict[str, object] = {}
    trace: list[dict[str, object]] | None = None
    for kind in ("mlp", "gru"):
        per_horizon: dict[str, object] = {}
        for horizon in HORIZONS:
            rows = build_rows(
                months, anomaly, observed, length=length, horizon=horizon, scale=scale
            )
            seed_scores: list[dict[str, float | None]] = []
            for seed in seeds:
                trained = fit_network(
                    rows.x,
                    rows.y,
                    rows.split == "train",
                    rows.split == "val",
                    kind=kind,
                    seed=seed,
                    quantiles=False,
                    learning_rate=learning_rate,
                )
                predicted = predict(trained, rows.x).reshape(-1)
                if kind == "mlp" and horizon == 1 and seed == seeds[0]:
                    trace = _trace(rows, predicted)
                entry: dict[str, float | None] = {}
                for split in ("val", "holdout"):
                    scored = _score_split(predicted, rows, split, scale)
                    entry[f"{split}_skill"] = None if scored is None else scored.skill
                    entry[f"{split}_mae"] = None if scored is None else scored.mae
                seed_scores.append(entry)
            per_horizon[str(horizon)] = _seed_summary(seed_scores)
        models[kind] = {
            "length": length,
            "learning_rate": learning_rate,
            "horizons": per_horizon,
        }
    quantile = _quantile_check(months, anomaly, observed, length, scale, learning_rate)
    return {
        "scale": scale,
        "baselines": _baseline_skills(baselines),
        "models": models,
        "quantile_coverage": quantile,
        "check_trace": trace,
    }


def _choose_settings(
    months: NDArray[np.datetime64],
    anomaly: NDArray[np.float64],
    observed: NDArray[np.bool_],
    scale: float,
) -> tuple[int, float]:
    """L 與學習率只看驗證期、種子 0 的 MLP。留出期不參與。"""
    best = (LENGTHS[0], LEARNING_RATES[0])
    best_skill = -1e9
    for length in LENGTHS:
        for learning_rate in LEARNING_RATES:
            rows = build_rows(
                months, anomaly, observed, length=length, horizon=1, scale=scale
            )
            trained = fit_network(
                rows.x,
                rows.y,
                rows.split == "train",
                rows.split == "val",
                kind="mlp",
                seed=0,
                quantiles=False,
                learning_rate=learning_rate,
            )
            scored = _score_split(
                predict(trained, rows.x).reshape(-1), rows, "val", scale
            )
            skill = -1e9 if scored is None or scored.skill is None else scored.skill
            if skill > best_skill:
                best_skill = skill
                best = (length, learning_rate)
    return best


def _trace(rows: Rows, predicted: NDArray[np.float64]) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for index in range(rows.y.shape[0]):
        if not rows.observed[index] or rows.split[index] == "train":
            continue
        points.append(
            {
                "month": str(rows.month[index]),
                "split": str(rows.split[index]),
                "actual": float(rows.y[index]),
                "predicted": float(predicted[index]),
            }
        )
    return points


def _quantile_check(
    months: NDArray[np.datetime64],
    anomaly: NDArray[np.float64],
    observed: NDArray[np.bool_],
    length: int,
    scale: float,
    learning_rate: float,
) -> dict[str, float]:
    rows = build_rows(months, anomaly, observed, length=length, horizon=1, scale=scale)
    trained = fit_network(
        rows.x,
        rows.y,
        rows.split == "train",
        rows.split == "val",
        kind="mlp",
        seed=0,
        quantiles=True,
        learning_rate=learning_rate,
    )
    predicted = predict(trained, rows.x)
    result: dict[str, float] = {}
    names = rows.split
    observed_rows = rows.observed
    actual = rows.y
    for split in ("val", "holdout"):
        chosen = (names == split) & observed_rows
        result[split] = coverage(
            predicted[chosen, 0], predicted[chosen, 2], actual[chosen]
        )
    return result


def _seed_summary(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for key in ("val_skill", "val_mae", "holdout_skill", "holdout_mae"):
        values: list[float] = []
        for row in rows:
            number = row[key]
            if isinstance(number, float):
                values.append(number)
        if not values:
            summary[key] = None
            summary[f"{key}_std"] = None
            continue
        summary[key] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def _baseline_skills(
    evaluated: dict[str, dict[int, dict[str, Score]]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    output: dict[str, dict[str, dict[str, float | None]]] = {}
    for split, label in (("validation", "val"), ("holdout", "holdout")):
        output[label] = {}
        for horizon, baselines in evaluated[split].items():
            by_name: dict[str, float | None] = {}
            skills: list[float] = []
            for name, score in baselines.items():
                if not isinstance(score, Score):
                    continue
                by_name[name] = score.skill
                if score.skill is not None:
                    skills.append(score.skill)
            output[label][str(horizon)] = {
                "strongest": None if not skills else float(max(skills)),
                **by_name,
            }
    return output


def beats_baseline(
    model_skill: float | None, model_std: float | None, baseline_skill: float | None
) -> bool:
    if model_skill is None or baseline_skill is None or model_std is None:
        return False
    return model_skill - baseline_skill > model_std


def run_forecasts(series: SeriesTable, observations) -> dict[str, object]:
    from trend_detective.nn.baselines import select_layer_series

    cities = world_city_ids(Path("data/seed/locations.world.json"))
    selected = select_layer_series(series, cities)
    records: list[dict[str, object]] = []
    for index in range(len(selected)):
        series_id = int(selected.series_id[index])
        mask = observations.series_id == series_id
        if not np.any(mask):
            continue
        variable_id = str(selected.variable_id[index])
        location_id = str(selected.location_id[index])
        forecast = series_forecast(
            observations.month[mask],
            observations.value_clean[mask],
            observations.fill_method[mask],
            observations.qc_flag[mask],
        )
        if (variable_id, location_id) not in CHECK_SERIES:
            forecast = {
                key: value for key, value in forecast.items() if key != "check_trace"
            }
        records.append(
            {
                "variable_id": variable_id,
                "location_id": location_id,
                "forecast": forecast,
            }
        )
    counted = _count_wins(records)
    return {
        "exploratory": True,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_series": len(records),
        "parameters": {
            "seeds": list(SEEDS),
            "lengths": list(LENGTHS),
            "learning_rates": list(LEARNING_RATES),
            "hidden": HIDDEN,
            "gru_hidden": GRU_HIDDEN,
            "epochs": EPOCHS,
            "horizons": list(HORIZONS),
            "variables": list(LAYER_VARIABLES),
            "locations": list(TAIWAN_POINTS + cities),
            "holdout": "evaluated once, after validation chose length and learning rate",
            "check_series": [list(item) for item in CHECK_SERIES],
            "check_series_rule": "fixed pairs spread across variables; not chosen by score",
        },
        "wins": counted,
        "series": records,
    }


def _count_wins(records: list[dict[str, object]]) -> dict[str, object]:
    wins: dict[str, dict[str, int]] = {"mlp": {}, "gru": {}}
    for kind in ("mlp", "gru"):
        for horizon in HORIZONS:
            wins[kind][str(horizon)] = 0
    for record in records:
        forecast = record["forecast"]
        assert isinstance(forecast, dict)
        baselines = forecast["baselines"]
        assert isinstance(baselines, dict)
        models = forecast["models"]
        assert isinstance(models, dict)
        for kind, payload in models.items():
            assert isinstance(payload, dict)
            horizons = payload["horizons"]
            assert isinstance(horizons, dict)
            for horizon, summary in horizons.items():
                assert isinstance(summary, dict)
                base = baselines["val"]
                assert isinstance(base, dict)
                horizon_base = base.get(horizon)
                assert isinstance(horizon_base, dict)
                if beats_baseline(
                    _as_float(summary.get("val_skill")),
                    _as_float(summary.get("val_skill_std")),
                    _as_float(horizon_base.get("strongest")),
                ):
                    wins[kind][horizon] += 1
    return {"validation_series_beating_strongest_baseline": wins}


def _as_float(value: object) -> float | None:
    if isinstance(value, float):
        return value
    return None


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    import argparse

    from trend_detective.nn.data import load_database

    parser = argparse.ArgumentParser(description="單一序列 MLP 與 GRU")
    parser.add_argument(
        "--output", type=Path, default=Path("data/analysis/nn/single_series.json")
    )
    args = parser.parse_args(argv)
    series, observations = load_database()
    payload = run_forecasts(series, observations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ready = _json_ready(payload)
    args.output.write_text(
        json.dumps(ready, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".run.json").write_text(
        json.dumps(
            {
                "command": "uv run python -m trend_detective.nn.single_series",
                "generated_at": payload["generated_at"],
                "exploratory": True,
                "holdout_evaluated_once": True,
                "n_series": payload["n_series"],
                "pretrained_model": None,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["wins"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
