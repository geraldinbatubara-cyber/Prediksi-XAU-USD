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
ACTUAL_HOURS = (8, 9, 10, 11, 12)

BASE_COLUMNS = [
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
    "estimated_direction",
    "status",
    "notes",
]

def hour_suffix(hour: int) -> str:
    return f"{hour:02d}00"


ACTUAL_COLUMNS = [
    field
    for hour in ACTUAL_HOURS
    for suffix in (hour_suffix(hour),)
    for field in (
        f"actual_timestamp_{suffix}",
        f"actual_open_{suffix}",
        f"actual_source_{suffix}",
        f"delta_{suffix}",
        f"delta_pct_{suffix}",
        f"actual_direction_{suffix}",
        f"direction_correct_{suffix}",
    )
]

MONITORING_COLUMNS = BASE_COLUMNS[:10] + ACTUAL_COLUMNS + BASE_COLUMNS[10:]


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

    legacy_to_0800 = {
        "actual_timestamp_wit": "actual_timestamp_0800",
        "actual_source": "actual_source_0800",
        "delta": "delta_0800",
        "delta_pct": "delta_pct_0800",
        "actual_direction": "actual_direction_0800",
        "direction_correct": "direction_correct_0800",
    }
    for old_column, new_column in legacy_to_0800.items():
        if old_column in frame.columns:
            missing = frame[new_column].isna() | (frame[new_column].astype(str) == "")
            frame.loc[missing, new_column] = frame.loc[missing, old_column]

    object_columns = [
        column
        for column in MONITORING_COLUMNS
        if not (
            column in {"reference_price", "estimate_tomorrow", "estimate_lower", "estimate_upper", "confidence"}
            or column.startswith("actual_open_")
            or column.startswith("delta_")
            or column.startswith("delta_pct_")
        )
    ]
    for column in object_columns:
        frame[column] = frame[column].astype("object")

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


def _format_hours(hours: list[int] | tuple[int, ...]) -> str:
    return ", ".join(f"{hour:02d}:00" for hour in hours)


def _is_blank(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _missing_actual_hours(row: pd.Series) -> list[int]:
    return [hour for hour in ACTUAL_HOURS if _is_blank(row.get(f"actual_open_{hour_suffix(hour)}", ""))]


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
        "estimated_direction": "Naik" if estimate >= latest else "Turun",
        "status": f"Menunggu aktual {_format_hours(ACTUAL_HOURS)} WIT",
        "notes": "Estimasi tersimpan otomatis dari data publik Yahoo Finance.",
    }
    for hour in ACTUAL_HOURS:
        suffix = hour_suffix(hour)
        row[f"actual_timestamp_{suffix}"] = ""
        row[f"actual_open_{suffix}"] = ""
        row[f"actual_source_{suffix}"] = ""
        row[f"delta_{suffix}"] = ""
        row[f"delta_pct_{suffix}"] = ""
        row[f"actual_direction_{suffix}"] = ""
        row[f"direction_correct_{suffix}"] = ""

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


def fetch_actual_at(target_date_wit: str, hour_wit: int) -> ActualOpen | None:
    target_at_wit = datetime.fromisoformat(f"{target_date_wit}T{hour_wit:02d}:00:00").replace(tzinfo=WIT)
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
        return None
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
        missing_hours = _missing_actual_hours(row)
        if not missing_hours:
            continue
        target_date = str(row["target_date_wit"])

        for hour in missing_hours:
            actual = fetch_actual_at(target_date, hour)
            if actual is None:
                continue

            suffix = hour_suffix(hour)
            estimate = float(row["estimate_tomorrow"])
            reference = float(row["reference_price"])
            delta = actual.price - estimate
            delta_pct = delta / estimate * 100
            actual_direction = "Naik" if actual.price >= reference else "Turun"
            direction_correct = str(row.get("estimated_direction", "")) == actual_direction
            frame.at[index, f"actual_timestamp_{suffix}"] = actual.timestamp_wit.isoformat(timespec="seconds")
            frame.at[index, f"actual_open_{suffix}"] = actual.price
            frame.at[index, f"actual_source_{suffix}"] = actual.source
            frame.at[index, f"delta_{suffix}"] = delta
            frame.at[index, f"delta_pct_{suffix}"] = delta_pct
            frame.at[index, f"actual_direction_{suffix}"] = actual_direction
            frame.at[index, f"direction_correct_{suffix}"] = direction_correct

        updated_missing_hours = _missing_actual_hours(frame.loc[index])
        if not updated_missing_hours:
            frame.at[index, "status"] = "Selesai"
            frame.at[index, "notes"] = (
                "Aktual diisi dari candle intraday terdekat pada/setelah "
                f"{_format_hours(ACTUAL_HOURS)} WIT."
            )
        elif len(updated_missing_hours) < len(ACTUAL_HOURS):
            frame.at[index, "status"] = "Sebagian selesai"
            frame.at[index, "notes"] = f"Menunggu candle intraday {_format_hours(updated_missing_hours)} WIT."
        else:
            frame.at[index, "status"] = f"Menunggu aktual {_format_hours(updated_missing_hours)} WIT"
            frame.at[index, "notes"] = f"Candle intraday {_format_hours(updated_missing_hours)} WIT belum tersedia."

    frame = frame.sort_values(["forecast_date_wit"], ascending=False)
    save_monitoring(frame, path)
    return frame


def monitoring_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hour in ACTUAL_HOURS:
        suffix = hour_suffix(hour)
        price = pd.to_numeric(frame[f"actual_open_{suffix}"], errors="coerce")
        completed = frame.loc[price.notna()].copy()
        if completed.empty:
            rows.append(
                {
                    "Jam WIT": f"{hour:02d}:00",
                    "Jumlah selesai": 0,
                    "MAE": np.nan,
                    "MAPE": np.nan,
                    "Akurasi arah": np.nan,
                    "Bias rata-rata": np.nan,
                }
            )
            continue

        delta = pd.to_numeric(completed[f"delta_{suffix}"], errors="coerce")
        delta_pct = pd.to_numeric(completed[f"delta_pct_{suffix}"], errors="coerce")
        direction = completed[f"direction_correct_{suffix}"].astype(str).str.lower().map({"true": True, "false": False})
        rows.append(
            {
                "Jam WIT": f"{hour:02d}:00",
                "Jumlah selesai": int(price.notna().sum()),
                "MAE": float(delta.abs().mean()),
                "MAPE": float(delta_pct.abs().mean()),
                "Akurasi arah": float(direction.mean() * 100) if direction.notna().any() else np.nan,
                "Bias rata-rata": float(delta.mean()),
            }
        )
    return pd.DataFrame(rows)
