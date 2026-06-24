from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from gold_forecast.data import load_market_data
from gold_forecast.model_v2 import train_model_v2
from gold_forecast.signals import build_signal


WIT = ZoneInfo("Asia/Jayapura")
UTC = ZoneInfo("UTC")
DATA_PATH = Path("data") / "monitoring.csv"
MODEL_NAME = "Model 2 - Lintas Pasar"

MONITORING_COLUMNS = [
    "forecast_date_wit",
    "target_date_wit",
    "forecast_timestamp_wit",
    "model",
    "reference_price",
    "estimate_tomorrow",
    "estimate_lower",
    "estimate_upper",
    "signal",
    "confidence",
    "actual_timestamp_wit",
    "actual_open_0800",
    "actual_source",
    "delta",
    "delta_pct",
    "estimated_direction",
    "actual_direction",
    "direction_correct",
    "status",
    "notes",
]


@dataclass
class ActualOpen:
    price: float
    timestamp_wit: datetime
    source: str


def now_wit() -> datetime:
    return datetime.now(WIT)


def load_monitoring(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=MONITORING_COLUMNS)
    frame = pd.read_csv(path)
    for column in MONITORING_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[MONITORING_COLUMNS]


def save_monitoring(frame: pd.DataFrame, path: Path = DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[MONITORING_COLUMNS].to_csv(path, index=False)


def _replace_row(frame: pd.DataFrame, row: dict[str, object]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame([row], columns=MONITORING_COLUMNS)
    same_date = frame["forecast_date_wit"].astype(str) == str(row["forecast_date_wit"])
    if same_date.any():
        frame.loc[same_date, MONITORING_COLUMNS] = pd.DataFrame([row], columns=MONITORING_COLUMNS).values
        return frame
    return pd.concat([frame, pd.DataFrame([row], columns=MONITORING_COLUMNS)], ignore_index=True)


def capture_estimate(captured_at: datetime | None = None, path: Path = DATA_PATH) -> pd.DataFrame:
    captured_at = captured_at or now_wit()
    forecast_date = captured_at.date()
    target_date = forecast_date + timedelta(days=1)

    market = load_market_data()
    result = train_model_v2(market)
    signal = build_signal(market, result.forecast)
    tomorrow = result.forecast.iloc[0]
    latest = float(market["gold"].dropna().iloc[-1])
    estimate = float(tomorrow["Estimasi"])

    row = {
        "forecast_date_wit": forecast_date.isoformat(),
        "target_date_wit": target_date.isoformat(),
        "forecast_timestamp_wit": captured_at.isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "reference_price": latest,
        "estimate_tomorrow": estimate,
        "estimate_lower": float(tomorrow["Batas bawah"]),
        "estimate_upper": float(tomorrow["Batas atas"]),
        "signal": signal.label,
        "confidence": signal.confidence,
        "actual_timestamp_wit": "",
        "actual_open_0800": "",
        "actual_source": "",
        "delta": "",
        "delta_pct": "",
        "estimated_direction": "Naik" if estimate >= latest else "Turun",
        "actual_direction": "",
        "direction_correct": "",
        "status": "Menunggu aktual 08:00 WIT",
        "notes": "Estimasi tersimpan otomatis dari data publik Yahoo Finance.",
    }

    frame = _replace_row(load_monitoring(path), row)
    frame = frame.sort_values(["forecast_date_wit"], ascending=False)
    save_monitoring(frame, path)
    return frame


def _normalize_intraday_index(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize(UTC)
    else:
        frame.index = frame.index.tz_convert(UTC)
    return frame.sort_index()


def fetch_actual_open_0800(target_date_wit: str) -> ActualOpen | None:
    target_at_wit = datetime.fromisoformat(f"{target_date_wit}T08:00:00").replace(tzinfo=WIT)
    target_at_utc = target_at_wit.astimezone(UTC)
    start = (target_at_utc - timedelta(hours=12)).date().isoformat()
    end = (target_at_utc + timedelta(days=1)).date().isoformat()
    data = yf.download(
        "GC=F",
        start=start,
        end=end,
        interval="5m",
        auto_adjust=True,
        progress=False,
        prepost=True,
    )
    if data.empty or "Open" not in data.columns:
        return None

    frame = _normalize_intraday_index(data)
    frame = frame[frame["Open"].notna()]
    if frame.empty:
        return None

    at_or_after = frame.loc[frame.index >= target_at_utc]
    if at_or_after.empty:
        candidate = frame.iloc[[-1]]
    else:
        candidate = at_or_after.iloc[[0]]

    timestamp_utc = candidate.index[0]
    if abs(timestamp_utc - target_at_utc) > pd.Timedelta(hours=2):
        return None

    return ActualOpen(
        price=float(candidate["Open"].iloc[0]),
        timestamp_wit=timestamp_utc.to_pydatetime().astimezone(WIT),
        source="GC=F 5m intraday open dari Yahoo Finance",
    )


def update_actuals(path: Path = DATA_PATH) -> pd.DataFrame:
    frame = load_monitoring(path)
    if frame.empty:
        save_monitoring(frame, path)
        return frame

    for index, row in frame.iterrows():
        if str(row.get("status", "")).startswith("Selesai"):
            continue
        target_date = str(row["target_date_wit"])
        actual = fetch_actual_open_0800(target_date)
        if actual is None:
            frame.at[index, "status"] = "Menunggu aktual 08:00 WIT"
            frame.at[index, "notes"] = "Candle intraday 08:00 WIT belum tersedia."
            continue

        estimate = float(row["estimate_tomorrow"])
        reference = float(row["reference_price"])
        delta = actual.price - estimate
        delta_pct = delta / estimate * 100
        actual_direction = "Naik" if actual.price >= reference else "Turun"
        direction_correct = str(row.get("estimated_direction", "")) == actual_direction
        frame.at[index, "actual_timestamp_wit"] = actual.timestamp_wit.isoformat(timespec="seconds")
        frame.at[index, "actual_open_0800"] = actual.price
        frame.at[index, "actual_source"] = actual.source
        frame.at[index, "delta"] = delta
        frame.at[index, "delta_pct"] = delta_pct
        frame.at[index, "actual_direction"] = actual_direction
        frame.at[index, "direction_correct"] = direction_correct
        frame.at[index, "status"] = "Selesai"
        frame.at[index, "notes"] = "Aktual diisi dari candle intraday terdekat pada/setelah 08:00 WIT."

    frame = frame.sort_values(["forecast_date_wit"], ascending=False)
    save_monitoring(frame, path)
    return frame


def monitoring_summary(frame: pd.DataFrame) -> dict[str, float]:
    completed = frame[frame["status"].astype(str).str.startswith("Selesai")].copy()
    if completed.empty:
        return {"count": 0, "mae": np.nan, "mape": np.nan, "direction_accuracy": np.nan}
    completed["delta"] = pd.to_numeric(completed["delta"], errors="coerce")
    completed["delta_pct"] = pd.to_numeric(completed["delta_pct"], errors="coerce")
    direction = completed["direction_correct"].astype(str).str.lower().map({"true": True, "false": False})
    return {
        "count": float(len(completed)),
        "mae": float(completed["delta"].abs().mean()),
        "mape": float(completed["delta_pct"].abs().mean()),
        "direction_accuracy": float(direction.mean() * 100) if direction.notna().any() else np.nan,
    }
