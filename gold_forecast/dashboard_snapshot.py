from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gold_forecast.direction_model import train_direction_model
from gold_forecast.model import train_and_forecast
from gold_forecast.model_v2 import train_model_v2


DASHBOARD_SNAPSHOT_VERSION = "dashboard-snapshot-v1"
DASHBOARD_SNAPSHOT_PATH = Path("data/precomputed/dashboard_snapshot.pkl")
V10_PARAMS_PATH = Path("data/precomputed/v10_params.json")


def build_dashboard_snapshot(
    market: pd.DataFrame,
    v10_leaderboard: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "version": DASHBOARD_SNAPSHOT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_last_date": pd.Timestamp(market.index.max()).isoformat(),
        "model_1": train_and_forecast(market["gold"]),
        "model_2": train_model_v2(market),
        "direction_model": train_direction_model(market),
        "v10_leaderboard": v10_leaderboard.head(1).copy(),
    }


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


def save_v10_params(leaderboard: pd.DataFrame, path: Path = V10_PARAMS_PATH) -> None:
    if leaderboard.empty:
        raise ValueError("Leaderboard Optimizer v10 kosong.")
    params = {str(key): _json_value(value) for key, value in leaderboard.iloc[0].items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, ensure_ascii=True), encoding="utf-8")


def load_v10_params(path: Path = V10_PARAMS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        params = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    return pd.DataFrame([params]) if isinstance(params, dict) else pd.DataFrame()

