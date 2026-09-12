import pandas as pd

from gold_forecast.dashboard_snapshot import dashboard_snapshot_is_current
from gold_forecast.forecast_validity import (
    compare_locked_prediction_to_live,
    completed_daily_frame,
    forecast_guard,
    live_quote_basis,
    rebase_forecast_to_live,
)


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


def test_live_quote_basis_accepts_fresh_quote() -> None:
    quote = pd.Series(
        {
            "bid": 4400.0,
            "ask": 4400.4,
            "received_at_utc": "2026-09-07T01:00:00Z",
            "clock_valid": True,
        }
    )
    result = live_quote_basis(quote, "2026-09-07T01:04:00Z")
    assert result["usable"] is True
    assert result["price"] == 4400.2


def test_live_quote_basis_rejects_stale_quote() -> None:
    quote = pd.Series(
        {"mid": 4400.2, "received_at_utc": "2026-09-07T01:00:00Z"}
    )
    assert live_quote_basis(quote, "2026-09-07T01:06:00Z")["usable"] is False


def test_rebase_forecast_preserves_projected_return() -> None:
    rebased = rebase_forecast_to_live(4480.0, 4400.0, 4450.0)
    assert abs(rebased - 4450.0 * (4480.0 / 4400.0)) < 1e-9


def test_model_3_daily_prediction_does_not_follow_live_quote() -> None:
    first_quote = compare_locked_prediction_to_live(4480.0, 4450.0)
    second_quote = compare_locked_prediction_to_live(4480.0, 4475.0)

    assert first_quote["prediction"] == 4480.0
    assert second_quote["prediction"] == 4480.0
    assert first_quote["live_difference"] == -30.0
    assert second_quote["live_difference"] == -5.0


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


def test_snapshot_price_revision_invalidates_forecast() -> None:
    result = forecast_guard(
        "2026-09-04",
        "2026-09-04",
        4429.80,
        pd.Series({"Batas bawah": 4300.0, "Batas atas": 4600.0}),
        source_price=4477.20,
    )

    assert result["code"] == "SOURCE_PRICE_MISMATCH"
    assert result["usable"] is False


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
        "market_last_price": 4513.50,
    }

    from gold_forecast.dashboard_snapshot import market_fingerprint

    current["market_fingerprint"] = market_fingerprint(market)

    assert dashboard_snapshot_is_current(stale, market, as_of) is False
    assert dashboard_snapshot_is_current(current, market, as_of) is True


def test_dashboard_snapshot_rebuilds_when_same_date_price_changes() -> None:
    from gold_forecast.dashboard_snapshot import market_fingerprint

    market = pd.DataFrame(
        {"gold": [4477.20]}, index=pd.to_datetime(["2026-09-04"])
    )
    snapshot = {
        "market_last_date": "2026-09-04",
        "market_feature_last_date": "2026-09-04",
        "market_last_price": 4429.80,
        "market_fingerprint": market_fingerprint(market),
    }
    assert dashboard_snapshot_is_current(
        snapshot,
        market,
        pd.Timestamp("2026-09-07 09:00:00", tz="Asia/Jayapura"),
    ) is False
