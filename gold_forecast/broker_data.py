from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd


BROKER_DATA_DIR = Path("data/broker")
BROKER_BARS_PATH = BROKER_DATA_DIR / "xauusd_m1.csv"
BROKER_QUOTE_PATH = BROKER_DATA_DIR / "latest_quote.csv"
BAR_COLUMNS = [
    "timestamp_utc",
    "broker_timestamp_utc",
    "clock_offset_hours",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread_points",
    "symbol",
    "source",
]
QUOTE_COLUMNS = [
    "timestamp_utc",
    "broker_timestamp_utc",
    "received_at_utc",
    "clock_offset_hours",
    "clock_valid",
    "bid",
    "ask",
    "mid",
    "spread",
    "symbol",
    "source",
]


def _read_csv(source: str | Path | IO[bytes] | IO[str] | pd.DataFrame | None) -> pd.DataFrame:
    if source is None:
        return pd.DataFrame()
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, Path) and not source.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(source)
    except Exception:
        return pd.DataFrame()


def _timestamp_column(frame: pd.DataFrame) -> str | None:
    aliases = ("timestamp_utc", "timestamp", "time", "datetime", "date")
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        if alias in lookup:
            return str(lookup[alias])
    return None


def load_broker_bars(
    source: str | Path | IO[bytes] | IO[str] | pd.DataFrame | None = BROKER_BARS_PATH,
) -> pd.DataFrame:
    frame = _read_csv(source)
    if frame.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    timestamp_column = _timestamp_column(frame)
    required = {"open", "high", "low", "close"}
    if timestamp_column is None or not required.issubset(frame.columns):
        return pd.DataFrame(columns=BAR_COLUMNS)

    frame = frame.rename(columns={timestamp_column: "timestamp_utc"})
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    frame["broker_timestamp_utc"] = frame["timestamp_utc"]
    frame["clock_offset_hours"] = 0.0
    for column in ["open", "high", "low", "close", "tick_volume", "spread_points"]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "symbol" not in frame.columns:
        frame["symbol"] = "XAUUSD"
    if "source" not in frame.columns:
        frame["source"] = "Broker CSV"

    frame = frame.dropna(subset=["timestamp_utc", "open", "high", "low", "close"])
    frame = frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    return frame[BAR_COLUMNS].reset_index(drop=True)


def load_broker_quote(
    source: str | Path | IO[bytes] | IO[str] | pd.DataFrame | None = BROKER_QUOTE_PATH,
) -> pd.DataFrame:
    frame = _read_csv(source)
    if frame.empty:
        return pd.DataFrame(columns=QUOTE_COLUMNS)

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    timestamp_column = _timestamp_column(frame)
    if timestamp_column is None or not {"bid", "ask"}.issubset(frame.columns):
        return pd.DataFrame(columns=QUOTE_COLUMNS)

    frame = frame.rename(columns={timestamp_column: "timestamp_utc"})
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    frame["broker_timestamp_utc"] = frame["timestamp_utc"]
    received_column = "received_at_utc" if "received_at_utc" in frame.columns else "updated_at"
    if received_column in frame.columns:
        frame["received_at_utc"] = pd.to_datetime(frame[received_column], errors="coerce", utc=True)
    else:
        frame["received_at_utc"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")

    skew_hours = (
        (frame["broker_timestamp_utc"] - frame["received_at_utc"]).dt.total_seconds() / 3600
    )
    rounded_offset = skew_hours.round()
    whole_hour_offset = (
        frame["received_at_utc"].notna()
        & ((skew_hours - rounded_offset).abs() <= (10 / 60))
        & (rounded_offset.abs() <= 14)
    )
    frame["clock_offset_hours"] = rounded_offset.where(whole_hour_offset, 0.0)
    frame["timestamp_utc"] = frame["broker_timestamp_utc"] - pd.to_timedelta(
        frame["clock_offset_hours"], unit="h"
    )
    normalized_skew_minutes = (
        (frame["timestamp_utc"] - frame["received_at_utc"]).dt.total_seconds().abs() / 60
    )
    frame["clock_valid"] = frame["received_at_utc"].isna() | (normalized_skew_minutes <= 10)
    frame["bid"] = pd.to_numeric(frame["bid"], errors="coerce")
    frame["ask"] = pd.to_numeric(frame["ask"], errors="coerce")
    frame["mid"] = (frame["bid"] + frame["ask"]) / 2
    frame["spread"] = frame["ask"] - frame["bid"]
    if "symbol" not in frame.columns:
        frame["symbol"] = "XAUUSD"
    if "source" not in frame.columns:
        frame["source"] = "Broker quote"

    frame = frame.dropna(subset=["timestamp_utc", "bid", "ask"])
    frame = frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    return frame[QUOTE_COLUMNS].reset_index(drop=True)


def apply_broker_clock_offset(
    bars: pd.DataFrame,
    quotes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bars.empty or quotes.empty or "clock_offset_hours" not in quotes.columns:
        return bars, quotes
    offset = pd.to_numeric(quotes.iloc[-1]["clock_offset_hours"], errors="coerce")
    if pd.isna(offset) or float(offset) == 0:
        return bars, quotes
    normalized = bars.copy()
    raw_timestamp = normalized.get("broker_timestamp_utc", normalized["timestamp_utc"])
    normalized["broker_timestamp_utc"] = pd.to_datetime(raw_timestamp, errors="coerce", utc=True)
    normalized["clock_offset_hours"] = float(offset)
    normalized["timestamp_utc"] = normalized["broker_timestamp_utc"] - pd.to_timedelta(float(offset), unit="h")
    normalized = normalized.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    return normalized[BAR_COLUMNS].reset_index(drop=True), quotes


def audit_broker_feed(
    bars: pd.DataFrame,
    quotes: pd.DataFrame,
    now: pd.Timestamp | None = None,
    stale_after_minutes: float = 5.0,
) -> dict[str, object]:
    now_utc = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")

    latest_quote = quotes.iloc[-1] if not quotes.empty else None
    latest_bar = bars.iloc[-1] if not bars.empty else None
    timestamps = []
    if latest_quote is not None:
        timestamps.append(pd.Timestamp(latest_quote["timestamp_utc"]))
    if latest_bar is not None:
        timestamps.append(pd.Timestamp(latest_bar["timestamp_utc"]))
    latest_timestamp = max(timestamps) if timestamps else pd.NaT
    latest_received_at = pd.NaT
    clock_offset_hours = 0.0
    clock_valid = True
    if latest_quote is not None:
        received_at = pd.Timestamp(latest_quote.get("received_at_utc", pd.NaT))
        if pd.notna(received_at):
            latest_received_at = received_at
        clock_offset_hours = float(latest_quote.get("clock_offset_hours", 0.0) or 0.0)
        clock_valid = bool(latest_quote.get("clock_valid", True))
    freshness_timestamp = latest_received_at if pd.notna(latest_received_at) else latest_timestamp
    age_minutes = (
        (now_utc - freshness_timestamp).total_seconds() / 60
        if pd.notna(freshness_timestamp)
        else np.nan
    )

    invalid_quotes = 0
    if not quotes.empty:
        invalid_quotes = int((quotes["ask"] < quotes["bid"]).sum())
    invalid_bars = 0
    if not bars.empty:
        invalid_bars = int(
            (
                (bars["high"] < bars[["open", "close", "low"]].max(axis=1))
                | (bars["low"] > bars[["open", "close", "high"]].min(axis=1))
            ).sum()
        )

    gaps_over_five = 0
    if len(bars) > 1:
        gaps_over_five = int((bars["timestamp_utc"].diff().dt.total_seconds() > 300).sum())

    return {
        "connected": bool(pd.notna(latest_timestamp)),
        "latest_timestamp": latest_timestamp,
        "latest_received_at": latest_received_at,
        "age_minutes": age_minutes,
        "stale": bool(
            pd.notna(age_minutes)
            and (age_minutes > stale_after_minutes or age_minutes < -1 or not clock_valid)
        ),
        "clock_offset_hours": clock_offset_hours,
        "clock_valid": clock_valid,
        "latest_quote": latest_quote,
        "latest_bar": latest_bar,
        "bar_rows": len(bars),
        "quote_rows": len(quotes),
        "invalid_quotes": invalid_quotes,
        "invalid_bars": invalid_bars,
        "gaps_over_five_minutes": gaps_over_five,
    }
