import numpy as np
import pandas as pd

import gold_forecast.live_trading as live_trading
import gold_forecast.sideways_moderate_live as moderate_live


def _daily_uptrend() -> pd.DataFrame:
    index = pd.bdate_range("2026-04-01", periods=90)
    close = pd.Series(np.linspace(3000.0, 4400.0, len(index)), index=index)
    return pd.DataFrame(
        {"Open": close - 2.0, "High": close + 5.0, "Low": close - 5.0, "Close": close},
        index=index,
    )


def _params() -> dict[str, object]:
    return {
        "Mode": "Trend", "Fast MA": 10, "Slow MA": 50,
        "Momentum hari": 10, "Threshold entry (%)": 0.15,
        "Lot": 0.01, "TP (USD)": 25.0, "SL (USD)": 10.0,
    }


def _broker_rows(day: pd.Timestamp) -> pd.DataFrame:
    timestamps = pd.date_range(day.tz_localize("UTC"), periods=24, freq="1h")
    price = np.linspace(4380.0, 4400.0, len(timestamps))
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps, "open": price, "high": price + 1.0,
            "low": price - 1.0, "close": price, "spread_points": 10.0,
        }
    )


def test_broker_valuation_prefers_fresh_bid_ask():
    now = pd.Timestamp("2026-08-24 12:00:00", tz="Asia/Jayapura")
    quote = pd.Series(
        {
            "bid": 4634.8,
            "ask": 4635.2,
            "timestamp_utc": now.tz_convert("UTC"),
            "received_at_utc": now.tz_convert("UTC"),
            "source": "MT5 DEMO",
        }
    )
    state = live_trading._broker_valuation_state(
        live_trading._broker_quote_state(quote, now), None, now
    )
    assert state["price"] == 4635.0
    assert state["buy_price"] == 4634.8
    assert state["sell_price"] == 4635.2
    assert not state["provisional"]


def test_broker_valuation_falls_back_to_fresh_m1_close():
    now = pd.Timestamp("2026-08-24 12:00:00", tz="Asia/Jayapura")
    quote = pd.Series(
        {
            "bid": 4608.5,
            "ask": 4609.1,
            "timestamp_utc": now.tz_convert("UTC") - pd.Timedelta(hours=4),
            "received_at_utc": now.tz_convert("UTC") - pd.Timedelta(hours=4),
            "source": "MT5 DEMO",
        }
    )
    bars = pd.DataFrame(
        {
            "timestamp_utc": [now.tz_convert("UTC") - pd.Timedelta(minutes=1)],
            "open": [4634.0],
            "high": [4636.0],
            "low": [4633.0],
            "close": [4635.0],
            "spread_points": [30.0],
        }
    )
    state = live_trading._broker_valuation_state(
        live_trading._broker_quote_state(quote, now), bars, now
    )
    assert state["available"]
    assert state["provisional"]
    assert state["price"] == 4635.0
    assert state["source"] == "MT5 Close M1 (provisional)"


def test_broker_valuation_rejects_stale_quote_and_stale_m1():
    now = pd.Timestamp("2026-08-24 12:00:00", tz="Asia/Jayapura")
    quote = pd.Series(
        {
            "bid": 4608.5,
            "ask": 4609.1,
            "timestamp_utc": now.tz_convert("UTC") - pd.Timedelta(hours=4),
            "received_at_utc": now.tz_convert("UTC") - pd.Timedelta(hours=4),
            "source": "MT5 DEMO",
        }
    )
    bars = pd.DataFrame(
        {
            "timestamp_utc": [now.tz_convert("UTC") - pd.Timedelta(minutes=10)],
            "open": [4608.0],
            "high": [4610.0],
            "low": [4607.0],
            "close": [4608.5],
            "spread_points": [30.0],
        }
    )
    state = live_trading._broker_valuation_state(
        live_trading._broker_quote_state(quote, now), bars, now
    )
    assert not state["available"]
    assert pd.isna(state["price"])


def test_fixed_delay_distinguishes_daily_signal_outside_m1_coverage():
    daily = _daily_uptrend()
    latest_day = pd.Timestamp(daily.index.max())
    signal, state = live_trading._fixed_delay_live_signal(
        daily,
        _broker_rows(latest_day),
        _params(),
        (latest_day + pd.Timedelta(days=1, hours=12)).tz_localize("UTC"),
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
        "P(trend)": 0.82, "P(BUY)": 0.76, "Direction confidence": 0.64,
        "Classifier BUY": True, "M15 alignment": True, "Regime": "BULLISH",
    }
    fixed_state = {"Status": "MENUNGGU SINYAL HARIAN", "Detail": "Syarat harian belum lengkap."}
    monkeypatch.setattr(live_trading, "_buy_specialist_observation", lambda *args, **kwargs: observation)
    monkeypatch.setattr(live_trading, "_fixed_delay_live_signal", lambda *args, **kwargs: (None, fixed_state))
    signal, state = live_trading._buy_specialist_v4_signal(
        daily, broker, _params(),
        pd.Timestamp("2026-08-06 12:00:00", tz="Asia/Jayapura"),
        pd.Timestamp("2026-07-24 16:00:00", tz="Asia/Jayapura"),
        {"model": "stub"}, live_trading._empty_ledger(),
    )
    assert signal is None
    assert state["Status"] == "MENUNGGU SINYAL HARIAN"
    assert state["Regime"] == "BULLISH"
    assert state["P(trend)"] == 0.82
    assert state["P(BUY)"] == 0.76


def test_historical_signal_is_not_active_on_newer_daily_candle():
    historical = {
        "signal_date": pd.Timestamp("2026-07-29"),
        "arah": "SELL",
    }
    assert (
        live_trading._signal_for_latest_evaluation(
            historical, pd.Timestamp("2026-08-09")
        )
        is None
    )
    active = live_trading._signal_for_latest_evaluation(
        historical, pd.Timestamp("2026-07-29")
    )
    assert active == historical
    assert active is not historical


def test_decision_code_reports_no_new_signal_instead_of_old_signal():
    code = live_trading._decision_code(
        None,
        {"signal_date": pd.Timestamp("2026-07-29"), "arah": "SELL"},
        {"Status trigger": "Menunggu sinyal Optimizer"},
        daily_data_stale=False,
        quote_configured=True,
        quote_fresh=True,
    )
    assert code == "NO_NEW_SIGNAL"


def test_stale_daily_data_has_priority_in_decision_code():
    code = live_trading._decision_code(
        None,
        None,
        {"Status trigger": "Menunggu sinyal Optimizer"},
        daily_data_stale=True,
        quote_configured=True,
        quote_fresh=False,
    )
    assert code == "DATA_STALE"


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


def test_new_position_id_stays_above_historical_event_maximum():
    signal = {
        "signal_date": pd.Timestamp("2026-08-18"),
        "arah": "BUY",
        "prediction": 4405.0,
        "reference_price": 4400.0,
        "expected_change_pct": 0.2,
        "source": "test",
    }
    opened = live_trading._maybe_open_position(
        live_trading._empty_ledger(),
        signal,
        {**_params(), "Max BUY": 1, "Max SELL": 1, "Max Total": 1},
        pd.Timestamp("2026-08-18 09:00:00", tz="Asia/Jayapura"),
        True,
        "Aktif",
        broker_ask=4400.2,
        historical_max_position_id=7,
    )
    assert opened.iloc[0]["position_id"] == 8


def test_pending_signal_is_promoted_to_open_when_execution_recovers():
    signal = {
        "signal_date": pd.Timestamp("2026-08-21"),
        "arah": "BUY",
        "prediction": 4610.0,
        "reference_price": 4600.0,
        "expected_change_pct": 0.22,
        "source": "test",
    }
    params = {**_params(), "Max BUY": 1, "Max SELL": 1, "Max Total": 1}
    detected_at = pd.Timestamp("2026-08-21 08:00:00", tz="Asia/Jayapura")
    pending = live_trading._maybe_open_position(
        live_trading._empty_ledger(),
        signal,
        params,
        detected_at,
        False,
        "Quote broker stale",
    )
    assert len(pending) == 1
    assert pending.iloc[0]["status"] == "SIGNAL"
    assert pd.isna(pending.iloc[0]["entry_price"])

    opened = live_trading._maybe_open_position(
        pending,
        signal,
        params,
        detected_at + pd.Timedelta(minutes=5),
        True,
        "Aktif",
        broker_ask=4601.25,
    )
    assert len(opened) == 1
    assert opened.iloc[0]["position_id"] == pending.iloc[0]["position_id"]
    assert opened.iloc[0]["detected_at_wit"] == pending.iloc[0]["detected_at_wit"]
    assert opened.iloc[0]["status"] == "OPEN"
    assert opened.iloc[0]["entry_price"] == 4601.25


def test_pending_signal_is_not_reported_as_already_executed():
    signal = {
        "signal_date": pd.Timestamp("2026-08-21"),
        "arah": "BUY",
        "prediction": 4610.0,
        "reference_price": 4600.0,
        "expected_change_pct": 0.22,
        "source": "test",
    }
    params = {**_params(), "Max BUY": 1, "Max SELL": 1, "Max Total": 1}
    pending = live_trading._maybe_open_position(
        live_trading._empty_ledger(),
        signal,
        params,
        pd.Timestamp("2026-08-21 08:00:00", tz="Asia/Jayapura"),
        False,
        "Quote broker stale",
    )
    state = live_trading._optimizer_trigger_state(
        pending, signal, params, False, "Quote broker stale"
    )
    assert not state["Sudah dieksekusi"]
    assert state["Status trigger"] == "Menunggu jam trading"
    assert state["Catatan"] == "Quote broker stale"


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


def _moderate_open_position(*, tp_usd=10.0, cl_usd=5.0, direction="BUY"):
    row = {column: "" for column in live_trading.LIVE_COLUMNS}
    row.update(
        {
            "position_id": 1,
            "status": "OPEN",
            "arah": direction,
            "lot": 0.01,
            "entry_time_wit": "2026-08-17 09:00:00 WIT",
            "entry_price": 4000.0,
            "tp_usd": tp_usd,
            "cl_usd": cl_usd,
            "swap": 0.0,
            "protection_mode": "Initial SL",
        }
    )
    return pd.DataFrame([row], columns=live_trading.LIVE_COLUMNS)


def _moderate_broker_path(peak=4008.0, finish=4006.0):
    timestamps = pd.date_range("2026-08-17 00:00:00", periods=120, freq="min", tz="UTC")
    first = np.linspace(4000.0, peak, 60)
    second = np.linspace(peak, finish, 60)
    close = np.concatenate([first, second])
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "open": close,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "spread_points": 10.0,
        }
    )


def test_moderate_regime_activates_break_even_at_one_r():
    managed = live_trading._manage_moderate_exit_protection(
        _moderate_open_position(tp_usd=15.0, cl_usd=5.0),
        _moderate_broker_path(peak=4006.0, finish=4005.5),
        bid=4005.5,
        ask=4005.7,
        now=pd.Timestamp("2026-08-17 11:00:00", tz="Asia/Jayapura"),
    )
    row = managed.iloc[0]
    assert bool(row["break_even_activated"])
    assert not bool(row["trailing_activated"])
    assert row["protection_mode"] == "Break-even"
    assert row["active_sl_price"] == 4000.0


def test_moderate_regime_activates_and_closes_at_atr_trailing():
    managed = live_trading._manage_moderate_exit_protection(
        _moderate_open_position(tp_usd=10.0, cl_usd=5.0),
        _moderate_broker_path(),
        bid=4006.0,
        ask=4006.2,
        now=pd.Timestamp("2026-08-17 11:00:00", tz="Asia/Jayapura"),
    )
    row = managed.iloc[0]
    assert bool(row["break_even_activated"])
    assert bool(row["trailing_activated"])
    assert row["protection_mode"] == "ATR trailing"
    assert row["active_sl_price"] > 4000.0
    closed = live_trading._close_hit_positions_quote(
        managed,
        bid=float(row["active_sl_price"]) - 0.01,
        ask=float(row["active_sl_price"]) + 0.19,
        now=pd.Timestamp("2026-08-17 11:01:00", tz="Asia/Jayapura"),
    )
    assert closed.iloc[0]["status"] == "CLOSED"
    assert closed.iloc[0]["exit_reason"] == "ATR trailing tersentuh"


def test_moderate_regime_sell_uses_ask_side_atr_trailing():
    managed = live_trading._manage_moderate_exit_protection(
        _moderate_open_position(direction="SELL"),
        _moderate_broker_path(peak=3992.0, finish=3994.0),
        bid=3994.0,
        ask=3994.2,
        now=pd.Timestamp("2026-08-17 11:00:00", tz="Asia/Jayapura"),
    )
    row = managed.iloc[0]
    assert bool(row["trailing_activated"])
    assert row["protection_mode"] == "ATR trailing"
    assert row["active_sl_price"] < 4000.0
    closed = live_trading._close_hit_positions_quote(
        managed,
        bid=float(row["active_sl_price"]) - 0.21,
        ask=float(row["active_sl_price"]) + 0.01,
        now=pd.Timestamp("2026-08-17 11:01:00", tz="Asia/Jayapura"),
    )
    assert closed.iloc[0]["status"] == "CLOSED"
    assert closed.iloc[0]["exit_reason"] == "ATR trailing tersentuh"


def test_moderate_regime_treats_no_opportunity_as_waiting(monkeypatch):
    def no_opportunity(*args, **kwargs):
        raise RuntimeError("Range detector tidak menghasilkan opportunity.")

    monkeypatch.setattr(moderate_live, "_mean_reversion_opportunities", no_opportunity)
    result = moderate_live._safe_live_opportunities(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 20.0
    )
    assert result.empty
