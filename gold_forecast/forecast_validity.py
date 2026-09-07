from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


WIT = ZoneInfo("Asia/Jayapura")


def completed_daily_frame(
    frame: pd.DataFrame,
    as_of: object,
) -> pd.DataFrame:
    """Return completed weekday rows before the current WIT calendar date."""
    if frame.empty:
        return frame.copy()

    timestamp = pd.Timestamp(as_of)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(WIT)
    else:
        timestamp = timestamp.tz_convert(WIT)
    current_wit_date = timestamp.tz_localize(None).normalize()

    index = pd.to_datetime(frame.index)
    if index.tz is not None:
        index = index.tz_convert(WIT).tz_localize(None)
    normalized_index = index.normalize()
    completed = normalized_index < current_wit_date
    weekday = normalized_index.dayofweek < 5
    return frame.loc[completed & weekday].copy()


def live_quote_basis(
    quote: pd.Series | None,
    as_of: object,
    max_age_minutes: float = 5.0,
) -> dict[str, object]:
    """Validate a broker quote before using it as an intraday forecast basis."""
    if quote is None:
        return {"usable": False, "price": np.nan, "timestamp": pd.NaT, "age_minutes": np.nan}

    bid = pd.to_numeric(quote.get("bid"), errors="coerce")
    ask = pd.to_numeric(quote.get("ask"), errors="coerce")
    mid = pd.to_numeric(quote.get("mid"), errors="coerce")
    price = float(mid) if pd.notna(mid) else (
        (float(bid) + float(ask)) / 2 if pd.notna(bid) and pd.notna(ask) else np.nan
    )
    received_at = quote.get("received_at_utc")
    timestamp_source = received_at if pd.notna(received_at) else quote.get("timestamp_utc")
    timestamp = pd.to_datetime(
        timestamp_source,
        errors="coerce",
        utc=True,
    )
    now = pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    age_minutes = (now - timestamp).total_seconds() / 60 if pd.notna(timestamp) else np.nan
    clock_valid = bool(quote.get("clock_valid", True))
    usable = bool(
        np.isfinite(price)
        and price > 0
        and pd.notna(age_minutes)
        and -1 <= age_minutes <= max_age_minutes
        and clock_valid
    )
    return {
        "usable": usable,
        "price": price,
        "timestamp": timestamp,
        "age_minutes": age_minutes,
    }


def rebase_forecast_to_live(
    estimate: object,
    daily_reference: object,
    live_price: object,
) -> float:
    """Preserve the model's projected return while moving its price basis live."""
    estimate_value = pd.to_numeric(estimate, errors="coerce")
    reference_value = pd.to_numeric(daily_reference, errors="coerce")
    live_value = pd.to_numeric(live_price, errors="coerce")
    if not all(np.isfinite(value) and value > 0 for value in [estimate_value, reference_value, live_value]):
        return float("nan")
    return float(live_value * estimate_value / reference_value)


def forecast_guard(
    source_date: object,
    completed_date: object,
    completed_price: float,
    forecast_row: pd.Series,
    source_price: float | None = None,
) -> dict[str, object]:
    source = pd.Timestamp(source_date).normalize()
    current = pd.Timestamp(completed_date).normalize()
    stale = source != current
    lower = float(forecast_row.get("Batas bawah", np.nan))
    upper = float(forecast_row.get("Batas atas", np.nan))
    outside_interval = (
        pd.notna(lower)
        and pd.notna(upper)
        and not lower <= completed_price <= upper
    )
    price_mismatch = (
        source_price is not None
        and pd.notna(source_price)
        and abs(float(source_price) - float(completed_price)) > 0.01
    )
    if stale:
        code = "STALE_SNAPSHOT"
        label = "Snapshot model tertinggal"
    elif price_mismatch:
        code = "SOURCE_PRICE_MISMATCH"
        label = "Harga sumber snapshot berubah"
    elif outside_interval:
        code = "OUT_OF_DISTRIBUTION"
        label = "Harga candle selesai di luar interval model"
    else:
        code = "VALID"
        label = "Valid berdasarkan candle selesai"
    return {
        "code": code,
        "label": label,
        "usable": code == "VALID",
        "outside_interval": outside_interval,
        "price_mismatch": price_mismatch,
        "age_days": max((current - source).days, 0),
    }
