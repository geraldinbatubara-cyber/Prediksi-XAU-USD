import pandas as pd

from gold_forecast.dashboard_snapshot import dashboard_snapshot_is_current
from gold_forecast.forecast_validity import completed_daily_frame, forecast_guard


def test_current_wit_daily_row_is_provisional() -> None:
    market = pd.DataFrame(
        {"gold": [4486.60, 4513.50]},
        index=pd.to_datetime(["2026-08-10", "2026-08-11"]),
    )

    completed = completed_daily_frame(
        market,
        pd.Timestamp("2026-08-11 12:00:00", tz="Asia/Jayapura"),
    )

    assert completed.index.tolist() == [pd.Timestamp("2026-08-10")]
    assert completed["gold"].iloc[-1] == 4486.60


def test_weekend_session_row_is_not_treated_as_completed_candle() -> None:
    market = pd.DataFrame(
        {"gold": [4680.60, 4666.60]},
        index=pd.to_datetime(["2026-08-21", "2026-08-23"]),
    )

    completed = completed_daily_frame(
        market,
        pd.Timestamp("2026-08-24 09:35:00", tz="Asia/Jayapura"),
    )

    assert completed.index.tolist() == [pd.Timestamp("2026-08-21")]
    assert completed["gold"].iloc[-1] == 4680.60


def test_friday_snapshot_remains_valid_when_sunday_row_is_provisional() -> None:
    market = pd.DataFrame(
        {"gold": [4680.60, 4666.60]},
        index=pd.to_datetime(["2026-08-21", "2026-08-23"]),
    )
    completed = completed_daily_frame(
        market,
        pd.Timestamp("2026-08-24 09:35:00", tz="Asia/Jayapura"),
    )
    latest_date = completed.index.max()
    latest_price = float(completed["gold"].iloc[-1])

    result = forecast_guard(
        "2026-08-21",
        latest_date,
        latest_price,
        pd.Series({"Batas bawah": 4600.0, "Batas atas": 4750.0}),
    )

    assert result["code"] == "VALID"
    assert result["usable"] is True


def test_snapshot_matches_latest_completed_candle() -> None:
    forecast = pd.Series({"Batas bawah": 4400.0, "Batas atas": 4550.0})

    result = forecast_guard(
        "2026-08-10",
        "2026-08-10",
        4486.60,
        forecast,
    )

    assert result["code"] == "VALID"
    assert result["usable"] is True
    assert result["label"] == "Valid berdasarkan candle selesai"


def test_snapshot_behind_completed_candle_is_stale() -> None:
    forecast = pd.Series({"Batas bawah": 4300.0, "Batas atas": 4550.0})

    result = forecast_guard(
        "2026-08-09",
        "2026-08-10",
        4486.60,
        forecast,
    )

    assert result["code"] == "STALE_SNAPSHOT"
    assert result["usable"] is False


def test_dashboard_snapshot_requires_latest_completed_candle() -> None:
    market = pd.DataFrame(
        {"gold": [4486.60, 4513.50]},
        index=pd.to_datetime(["2026-08-10", "2026-08-11"]),
    )
    as_of = pd.Timestamp("2026-08-12 08:00:00", tz="Asia/Jayapura")

    stale = {
        "market_last_date": "2026-08-10",
        "market_feature_last_date": "2026-08-10",
    }
    current = {
        "market_last_date": "2026-08-11",
        "market_feature_last_date": "2026-08-11",
    }

    assert dashboard_snapshot_is_current(stale, market, as_of) is False
    assert dashboard_snapshot_is_current(current, market, as_of) is True
