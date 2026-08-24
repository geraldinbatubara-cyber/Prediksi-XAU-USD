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


def forecast_guard(
    source_date: object,
    completed_date: object,
    completed_price: float,
    forecast_row: pd.Series,
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
    if stale:
        code = "STALE_SNAPSHOT"
        label = "Snapshot model tertinggal"
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
        "age_days": max((current - source).days, 0),
    }
