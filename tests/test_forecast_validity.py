import pandas as pd

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
