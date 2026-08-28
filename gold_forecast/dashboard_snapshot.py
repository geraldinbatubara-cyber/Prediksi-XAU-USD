from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from gold_forecast.direction_model import train_direction_model
from gold_forecast.forecast_validity import completed_daily_frame
from gold_forecast.model import train_and_forecast
from gold_forecast.model_v2 import train_model_v2
from gold_forecast.model_v2 import _market_features


DASHBOARD_SNAPSHOT_VERSION = "dashboard-snapshot-v2-v1-only"
DASHBOARD_SNAPSHOT_PATH = Path("data/precomputed/dashboard_snapshot.pkl")
V1_PARAMS_PATH = Path("data/precomputed/v1_params.json")


def build_dashboard_snapshot(
    market: pd.DataFrame,
    v1_leaderboard: pd.DataFrame,
    generated_at: object | None = None,
) -> dict[str, Any]:
    generated_at = (
        pd.Timestamp.now(tz="UTC")
        if generated_at is None
        else pd.Timestamp(generated_at)
    )
    if generated_at.tzinfo is None:
        generated_at = generated_at.tz_localize("UTC")
    else:
        generated_at = generated_at.tz_convert("UTC")
    completed_market = completed_daily_frame(market, generated_at)
    if completed_market.empty:
        raise ValueError("Tidak ada candle harian selesai untuk membangun snapshot.")
    market = completed_market
    complete_features = _market_features(market).dropna()
    return {
        "version": DASHBOARD_SNAPSHOT_VERSION,
        "generated_at_utc": generated_at.isoformat(),
        "market_last_date": pd.Timestamp(market.index.max()).isoformat(),
        "market_last_price": float(market["gold"].iloc[-1]),
        "market_training_min": float(market["gold"].min()),
        "market_training_max": float(market["gold"].max()),
        "market_feature_last_date": (
            pd.Timestamp(complete_features.index.max()).isoformat()
            if not complete_features.empty
            else None
        ),
        "model_1": train_and_forecast(
            market["gold"], evaluate_walk_forward=True
        ),
        "model_2": train_model_v2(market, evaluate_walk_forward=True),
        "direction_model": train_direction_model(market),
        "v1_leaderboard": v1_leaderboard.head(1).copy(),
    }


def dashboard_snapshot_is_current(
    snapshot: dict[str, Any] | None,
    market: pd.DataFrame,
    as_of: object,
) -> bool:
    if not isinstance(snapshot, dict):
        return False

    completed_market = completed_daily_frame(market, as_of)
    if completed_market.empty:
        return False
    expected_date = pd.Timestamp(completed_market.index.max()).normalize()

    for key in ("market_last_date", "market_feature_last_date"):
        value = snapshot.get(key)
        if value is None:
            return False
        try:
            source_date = pd.Timestamp(value).normalize()
        except (TypeError, ValueError):
            return False
        if source_date != expected_date:
            return False
    return True


def save_dashboard_snapshot(
    snapshot: dict[str, Any],
    path: Path = DASHBOARD_SNAPSHOT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(snapshot, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_dashboard_snapshot(
    path: Path = DASHBOARD_SNAPSHOT_PATH,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as file:
            snapshot = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None
    if not isinstance(snapshot, dict) or snapshot.get("version") != DASHBOARD_SNAPSHOT_VERSION:
        return None
    return snapshot


def _json_value(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def save_v1_params(leaderboard: pd.DataFrame, path: Path = V1_PARAMS_PATH) -> None:
    if leaderboard.empty:
        raise ValueError("Leaderboard Optimizer v1 kosong.")
    params = {str(key): _json_value(value) for key, value in leaderboard.iloc[0].items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, ensure_ascii=True), encoding="utf-8")


def load_v1_params(path: Path = V1_PARAMS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        params = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    return pd.DataFrame([params]) if isinstance(params, dict) else pd.DataFrame()

