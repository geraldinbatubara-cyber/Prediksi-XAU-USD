import numpy as np
import pandas as pd

import gold_forecast.live_trading as live_trading


def _daily_uptrend() -> pd.DataFrame:
    index = pd.bdate_range("2026-04-01", periods=90)
    close = pd.Series(np.linspace(3000.0, 4400.0, len(index)), index=index)
    return pd.DataFrame(
        {
            "Open": close - 2.0,
            "High": close + 5.0,
            "Low": close - 5.0,
            "Close": close,
        },
        index=index,
    )


def _params() -> dict[str, object]:
    return {
        "Mode": "Trend",
        "Fast MA": 10,
        "Slow MA": 50,
        "Momentum hari": 10,
        "Threshold entry (%)": 0.15,
        "Lot": 0.01,
        "TP (USD)": 25.0,
        "SL (USD)": 10.0,
    }


def _broker_rows(day: pd.Timestamp) -> pd.DataFrame:
    timestamps = pd.date_range(
        day.tz_localize("UTC"), periods=24, freq="1h"
    )
    price = np.linspace(4380.0, 4400.0, len(timestamps))
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "spread_points": 10.0,
        }
    )


def test_fixed_delay_distinguishes_daily_signal_outside_m1_coverage():
    daily = _daily_uptrend()
    latest_day = pd.Timestamp(daily.index.max())
    broker = _broker_rows(latest_day)
    now = (latest_day + pd.Timedelta(days=1, hours=12)).tz_localize("UTC")

    signal, state = live_trading._fixed_delay_live_signal(
        daily,
        broker,
        _params(),
        now,
        pd.Timestamp("2026-07-23", tz="Asia/Jayapura"),
    )

    assert signal is None
    assert state["Kode diagnosis"] == "DAILY_SIGNAL_OUTSIDE_M1_COVERAGE"
    assert state["Jumlah sinyal harian sejak aktivasi"] > 0
    assert state["Status sinyal harian"] == "Kondisi BUY siap"


def test_buy_specialist_keeps_observation_when_fixed_delay_waits(monkeypatch):
    daily = _daily_uptrend()
    broker = _broker_rows(pd.Timestamp(daily.index.max()))
    observation = {
        "Evaluation time": pd.Timestamp("2026-08-05 23:00:00"),
        "P(trend)": 0.82,
        "P(BUY)": 0.76,
        "Direction confidence": 0.64,
        "Classifier BUY": True,
        "M15 alignment": True,
        "Regime": "BULLISH",
    }
    fixed_state = {
        "Status": "MENUNGGU SINYAL HARIAN",
        "Detail": "Syarat harian belum lengkap.",
    }
    monkeypatch.setattr(
        live_trading,
        "_buy_specialist_observation",
        lambda *args, **kwargs: observation,
    )
    monkeypatch.setattr(
        live_trading,
        "_fixed_delay_live_signal",
        lambda *args, **kwargs: (None, fixed_state),
    )

    signal, state = live_trading._buy_specialist_v4_signal(
        daily,
        broker,
        _params(),
        pd.Timestamp("2026-08-06 12:00:00", tz="Asia/Jayapura"),
        pd.Timestamp("2026-07-24 16:00:00", tz="Asia/Jayapura"),
        {"model": "stub"},
        live_trading._empty_ledger(),
    )

    assert signal is None
    assert state["Status"] == "MENUNGGU SINYAL HARIAN"
    assert state["Regime"] == "BULLISH"
    assert state["P(trend)"] == 0.82
    assert state["P(BUY)"] == 0.76
def test_intraday_signals_keep_timestamp_and_dynamic_barriers():
    params = {
        **_params(),
        "Max BUY": 2,
        "Max SELL": 2,
        "Max Total": 2,
    }
    now = pd.Timestamp("2026-08-17 12:00:00", tz="Asia/Jayapura")
    base = {
        "prediction": 4006.4,
        "reference_price": 4000.0,
        "expected_change_pct": 0.16,
        "arah": "BUY",
        "source": "Moderate Regime Sideways v9",
        "tp_usd": 8.0,
        "sl_usd": 6.0,
        "intraday_signal": True,
    }
    first = live_trading._maybe_open_position(
        live_trading._empty_ledger(),
        {**base, "signal_date": pd.Timestamp("2026-08-17 03:00:00")},
        params,
        now,
        True,
        "Aktif",
        broker_ask=4000.2,
    )
    second = live_trading._maybe_open_position(
        first,
        {**base, "signal_date": pd.Timestamp("2026-08-17 03:15:00")},
        params,
        now,
        True,
        "Aktif",
        broker_ask=4000.4,
    )
    assert len(second) == 2
    assert second["signal_date"].tolist() == [
        "2026-08-17 03:00:00",
        "2026-08-17 03:15:00",
    ]
    assert second.iloc[0]["tp_usd"] == 8.0
    assert second.iloc[0]["cl_usd"] == 6.0


def test_moderate_regime_closes_at_twelve_hour_time_stop():
    row = {
        column: "" for column in live_trading.LIVE_COLUMNS
    }
    row.update(
        {
            "position_id": 1,
            "status": "OPEN",
            "arah": "BUY",
            "lot": 0.01,
            "entry_time_wit": "2026-08-17 08:00:00 WIT",
            "entry_price": 4000.0,
            "swap": 0.0,
        }
    )
    closed = live_trading._close_time_stop_positions_quote(
        pd.DataFrame([row], columns=live_trading.LIVE_COLUMNS),
        4003.0,
        4003.2,
        pd.Timestamp("2026-08-17 20:01:00", tz="Asia/Jayapura"),
    )
    assert closed.iloc[0]["status"] == "CLOSED"
    assert closed.iloc[0]["exit_reason"] == "Time stop 12 jam"
