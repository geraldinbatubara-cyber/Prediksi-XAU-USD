from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_forecast.simulation import CONTRACT_OUNCES_PER_LOT, SimulationResult, _result


OPTIMIZATION_START = pd.Timestamp("2025-01-01")
OPTIMIZATION_END = pd.Timestamp("2026-06-30")
V10_WALK_FORWARD_START = pd.Timestamp("2023-01-01")
REAL_DATA_TEST_START = pd.Timestamp("2026-07-01")
REAL_DATA_TEST_END = pd.Timestamp("2026-07-16")
INITIAL_EQUITY = 1000.0
PHASE_GROWTH = 0.20
BUY_SWAP_PER_001_LOT = 0.2
SELL_SWAP_PER_001_LOT = 0.0
LIVE_REENTRY_BUFFER_USD = 3.0


@dataclass
class DynamicPosition:
    position_id: int
    phase: int
    model_name: str
    strategy_name: str
    signal_date: pd.Timestamp
    direction: str
    lot_size: float
    confidence: float
    entry_price: float
    prediction: float
    expected_change_pct: float
    take_profit_usd: float
    stop_loss_usd: float | None
    profit_close_usd: float | None
    profit_protection_activation_usd: float | None
    profit_protection_floor_usd: float | None
    profit_protection_trail_usd: float | None
    entry_threshold_pct: float
    max_positions: int
    swap_paid: float = 0.0
    peak_profit_usd: float = 0.0


@dataclass
class MultiPhaseSimulationResult:
    summary: dict[str, float]
    phases: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr_pct(gold_ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    high = gold_ohlc["High"].astype(float)
    low = gold_ohlc["Low"].astype(float)
    close = gold_ohlc["Close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean() / close * 100


def _indicator_predictions(
    gold_ohlc: pd.DataFrame,
    mode: str,
    fast_window: int,
    slow_window: int,
    momentum_days: int,
    entry_threshold_pct: float,
    test_start: pd.Timestamp = OPTIMIZATION_START,
    test_end: pd.Timestamp = OPTIMIZATION_END,
) -> pd.Series:
    close = gold_ohlc["Close"].astype(float)
    fast_ma = close.rolling(fast_window).mean()
    slow_ma = close.rolling(slow_window).mean()
    momentum = close.pct_change(momentum_days) * 100
    atr_pct = _atr_pct(gold_ohlc)
    rsi = _rsi(close)
    previous_high = gold_ohlc["High"].astype(float).rolling(slow_window).max().shift(1)
    previous_low = gold_ohlc["Low"].astype(float).rolling(slow_window).min().shift(1)

    expected_change = pd.Series(0.0, index=gold_ohlc.index)
    signal_size = (atr_pct * 0.75).clip(lower=entry_threshold_pct + 0.05, upper=1.8)

    if mode == "Trend":
        buy = (close > fast_ma) & (fast_ma > slow_ma) & (momentum > entry_threshold_pct)
        sell = (close < fast_ma) & (fast_ma < slow_ma) & (momentum < -entry_threshold_pct)
    elif mode == "Breakout":
        buy = (close > previous_high) & (momentum > 0)
        sell = (close < previous_low) & (momentum < 0)
    elif mode == "Pullback":
        buy = (close > slow_ma) & (rsi < 42) & (momentum > -entry_threshold_pct)
        sell = (close < slow_ma) & (rsi > 58) & (momentum < entry_threshold_pct)
    else:
        buy = pd.Series(False, index=gold_ohlc.index)
        sell = pd.Series(False, index=gold_ohlc.index)

    expected_change = expected_change.mask(buy, signal_size)
    expected_change = expected_change.mask(sell, -signal_size)
    predictions = close * (1 + expected_change / 100)
    predictions = predictions.loc[(predictions.index >= test_start) & (predictions.index <= test_end)]
    predictions = predictions[expected_change.loc[predictions.index].abs() >= entry_threshold_pct]
    return predictions.dropna()


def _fixed_lot_signals(predictions: pd.Series, lot_size: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction": predictions,
            "lot_size": lot_size,
            "confidence": np.nan,
        },
        index=predictions.index,
    ).dropna(subset=["prediction"])


def _indicator_predictions_v2(
    gold_ohlc: pd.DataFrame,
    mode: str,
    fast_window: int,
    slow_window: int,
    momentum_days: int,
    entry_threshold_pct: float,
    confidence_cutoff: float,
    test_start: pd.Timestamp = OPTIMIZATION_START,
    test_end: pd.Timestamp = OPTIMIZATION_END,
) -> pd.DataFrame:
    close = gold_ohlc["Close"].astype(float)
    high = gold_ohlc["High"].astype(float)
    low = gold_ohlc["Low"].astype(float)
    fast_ma = close.rolling(fast_window).mean()
    slow_ma = close.rolling(slow_window).mean()
    momentum = close.pct_change(momentum_days) * 100
    atr_pct = _atr_pct(gold_ohlc)
    rsi = _rsi(close)
    previous_high = high.rolling(slow_window).max().shift(1)
    previous_low = low.rolling(slow_window).min().shift(1)
    atr_baseline = atr_pct.rolling(80).median().replace(0, np.nan)

    if mode == "Hybrid Trend":
        buy = (close > fast_ma) & (fast_ma > slow_ma) & (momentum > entry_threshold_pct)
        sell = (close < fast_ma) & (fast_ma < slow_ma) & (momentum < -entry_threshold_pct)
        alignment = ((fast_ma / slow_ma - 1).abs() * 100).fillna(0)
    elif mode == "Volatility Breakout":
        strong_volatility = atr_pct > atr_baseline
        buy = (close > previous_high) & (momentum > 0) & strong_volatility
        sell = (close < previous_low) & (momentum < 0) & strong_volatility
        alignment = (atr_pct / atr_baseline).replace([np.inf, -np.inf], np.nan).fillna(0)
    elif mode == "Pullback Confirm":
        buy = (close > slow_ma) & (rsi < 45) & (momentum > -entry_threshold_pct / 2)
        sell = (close < slow_ma) & (rsi > 55) & (momentum < entry_threshold_pct / 2)
        alignment = (abs(50 - rsi) / 50).fillna(0)
    else:
        buy = pd.Series(False, index=gold_ohlc.index)
        sell = pd.Series(False, index=gold_ohlc.index)
        alignment = pd.Series(0.0, index=gold_ohlc.index)

    signal_direction = pd.Series(0.0, index=gold_ohlc.index)
    signal_direction = signal_direction.mask(buy, 1.0)
    signal_direction = signal_direction.mask(sell, -1.0)
    momentum_strength = (momentum.abs() / max(entry_threshold_pct, 0.01)).clip(0, 4)
    volatility_strength = (atr_pct / atr_baseline).replace([np.inf, -np.inf], np.nan).clip(0, 3).fillna(0)
    alignment_strength = alignment.clip(0, 3)
    confidence = ((momentum_strength * 0.45) + (volatility_strength * 0.30) + (alignment_strength * 0.25)) / 4
    confidence = confidence.clip(0, 1)
    expected_change = signal_direction * (entry_threshold_pct + 0.25 + confidence * 1.35)
    lot_size = np.where(confidence >= confidence_cutoff, 0.02, 0.01)

    signals = pd.DataFrame(
        {
            "prediction": close * (1 + expected_change / 100),
            "lot_size": lot_size,
            "confidence": confidence * 100,
        },
        index=gold_ohlc.index,
    )
    signals = signals.loc[(signals.index >= test_start) & (signals.index <= test_end)]
    signals = signals[signal_direction.loc[signals.index] != 0]
    signals = signals[expected_change.loc[signals.index].abs() >= entry_threshold_pct]
    return signals.dropna()


def _strategy_score(summary: dict[str, float]) -> tuple[float, float, float, float, float]:
    return (
        float(summary["Fase selesai"]),
        float(summary["Equity akhir"]),
        -float(summary["Max drawdown"]),
        float(summary["Profit factor"]) if not pd.isna(summary["Profit factor"]) else 0.0,
        float(summary["Jumlah transaksi"]),
    )


def _swap_cost(position: DynamicPosition) -> float:
    swap_per_001 = BUY_SWAP_PER_001_LOT if position.direction == "BUY" else SELL_SWAP_PER_001_LOT
    return swap_per_001 * (position.lot_size / 0.01)


def _dynamic_unrealized(position: DynamicPosition, price: float) -> float:
    units = position.lot_size * CONTRACT_OUNCES_PER_LOT
    if position.direction == "BUY":
        return (price - position.entry_price) * units
    return (position.entry_price - price) * units


def _dynamic_close_row(
    position: DynamicPosition,
    exit_date: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    cash_balance: float,
) -> dict[str, object]:
    gross_pnl = _dynamic_unrealized(position, exit_price)
    net_pnl = gross_pnl - position.swap_paid
    return {
        "Fase": position.phase,
        "Model": position.model_name,
        "Strategi": position.strategy_name,
        "Position ID": position.position_id,
        "Tanggal sinyal": position.signal_date,
        "Waktu sinyal": "23:59 WIT",
        "Tanggal entry": position.signal_date,
        "Waktu entry": "23:59 WIT",
        "Tanggal tutup": exit_date,
        "Waktu tutup": "Saat TP/SL tersentuh" if exit_reason in {"TP tersentuh", "SL tersentuh"} else "Close harian",
        "Arah": position.direction,
        "Lot": position.lot_size,
        "Confidence": position.confidence,
        "Prediksi": position.prediction,
        "Expected change (%)": position.expected_change_pct,
        "Entry": position.entry_price,
        "Exit": exit_price,
        "Alasan exit": exit_reason,
        "TP (USD)": position.take_profit_usd,
        "SL (USD)": np.nan if position.stop_loss_usd is None else position.stop_loss_usd,
        "Floating profit close (USD)": np.nan if position.profit_close_usd is None else position.profit_close_usd,
        "Profit protection aktif (USD)": (
            np.nan if position.profit_protection_activation_usd is None else position.profit_protection_activation_usd
        ),
        "Profit protection floor (USD)": (
            np.nan if position.profit_protection_floor_usd is None else position.profit_protection_floor_usd
        ),
        "Profit protection trail (USD)": (
            np.nan if position.profit_protection_trail_usd is None else position.profit_protection_trail_usd
        ),
        "Peak floating profit (USD)": position.peak_profit_usd,
        "Threshold entry (%)": position.entry_threshold_pct,
        "Gross P/L": gross_pnl,
        "Swap": -position.swap_paid,
        "Net P/L": net_pnl,
        "Balance": cash_balance,
        "Batas posisi": position.max_positions,
    }


def _simulate_phase(
    signals: pd.DataFrame,
    gold_ohlc: pd.DataFrame,
    model_name: str,
    *,
    phase: int,
    initial_balance: float,
    target_equity: float,
    take_profit_usd: float,
    max_buy_positions: int = 8,
    max_sell_positions: int = 10,
    entry_threshold_pct: float = 0.15,
    stop_loss_usd: float | None = 10.0,
    strategy_name: str,
    live_rules: bool = False,
    risk_cap_pct: float | None = None,
    profit_close_usd: float | None = None,
    profit_protection_activation_usd: float | None = None,
    profit_protection_floor_usd: float | None = None,
    profit_protection_trail_usd: float | None = None,
    close_on_target_equity: bool = True,
    accrue_swap_by_elapsed_days: bool = False,
) -> SimulationResult:
    cash_balance = initial_balance
    next_position_id = 1
    closed_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    open_positions: list[DynamicPosition] = []
    last_swap_dates: dict[int, pd.Timestamp] = {}
    last_cl_price: dict[str, float] = {}
    if signals.empty or gold_ohlc.empty:
        return _result(closed_rows, equity_rows, initial_balance, target_equity)

    signal_dates = set(signals.index)

    for current_date, candle in gold_ohlc.iterrows():
        if current_date < signals.index.min():
            continue

        high = float(candle["High"])
        low = float(candle["Low"])
        close = float(candle["Close"])

        still_open: list[DynamicPosition] = []
        for position in open_positions:
            units = position.lot_size * CONTRACT_OUNCES_PER_LOT
            effective_profit_usd = position.profit_close_usd if position.profit_close_usd is not None else position.take_profit_usd
            take_profit_points = None if position.profit_protection_activation_usd is not None else effective_profit_usd / units
            stop_loss_points = None if position.stop_loss_usd is None else position.stop_loss_usd / units
            profit_exit_reason = (
                f"Floating profit >= USD {effective_profit_usd:g}"
                if position.profit_close_usd is not None
                else "TP tersentuh"
            )
            if position.direction == "BUY":
                favorable_profit = _dynamic_unrealized(position, high)
                adverse_profit = _dynamic_unrealized(position, low)
                position.peak_profit_usd = max(position.peak_profit_usd, favorable_profit)
                tp_price = None if take_profit_points is None else position.entry_price + take_profit_points
                sl_price = None if stop_loss_points is None else position.entry_price - stop_loss_points
                hit_tp = tp_price is not None and high >= tp_price
                hit_sl = sl_price is not None and low <= sl_price
                if hit_sl:
                    exit_price = sl_price
                    exit_reason = "SL tersentuh"
                elif hit_tp:
                    exit_price = tp_price
                    exit_reason = profit_exit_reason
                elif (
                    position.profit_protection_activation_usd is not None
                    and position.peak_profit_usd >= position.profit_protection_activation_usd
                ):
                    locked_profit = max(
                        float(position.profit_protection_floor_usd or 0.0),
                        position.peak_profit_usd - float(position.profit_protection_trail_usd or 0.0),
                    )
                    if adverse_profit <= locked_profit:
                        exit_price = position.entry_price + (locked_profit / units)
                        exit_reason = (
                            f"Profit protection: peak USD {position.peak_profit_usd:g}, "
                            f"lock USD {locked_profit:g}"
                        )
                    else:
                        still_open.append(position)
                        continue
                else:
                    still_open.append(position)
                    continue
            else:
                favorable_profit = _dynamic_unrealized(position, low)
                adverse_profit = _dynamic_unrealized(position, high)
                position.peak_profit_usd = max(position.peak_profit_usd, favorable_profit)
                tp_price = None if take_profit_points is None else position.entry_price - take_profit_points
                sl_price = None if stop_loss_points is None else position.entry_price + stop_loss_points
                hit_tp = tp_price is not None and low <= tp_price
                hit_sl = sl_price is not None and high >= sl_price
                if hit_sl:
                    exit_price = sl_price
                    exit_reason = "SL tersentuh"
                elif hit_tp:
                    exit_price = tp_price
                    exit_reason = profit_exit_reason
                elif (
                    position.profit_protection_activation_usd is not None
                    and position.peak_profit_usd >= position.profit_protection_activation_usd
                ):
                    locked_profit = max(
                        float(position.profit_protection_floor_usd or 0.0),
                        position.peak_profit_usd - float(position.profit_protection_trail_usd or 0.0),
                    )
                    if adverse_profit <= locked_profit:
                        exit_price = position.entry_price - (locked_profit / units)
                        exit_reason = (
                            f"Profit protection: peak USD {position.peak_profit_usd:g}, "
                            f"lock USD {locked_profit:g}"
                        )
                    else:
                        still_open.append(position)
                        continue
                else:
                    still_open.append(position)
                    continue

            cash_balance += _dynamic_unrealized(position, exit_price)
            closed_rows.append(_dynamic_close_row(position, current_date, exit_price, exit_reason, cash_balance))
            if live_rules and exit_reason == "SL tersentuh":
                last_cl_price[position.direction] = exit_price
        open_positions = still_open

        for position in open_positions:
            daily_swap = _swap_cost(position)
            if accrue_swap_by_elapsed_days:
                current_day = pd.Timestamp(current_date).normalize()
                previous_day = last_swap_dates.get(
                    position.position_id,
                    pd.Timestamp(position.signal_date).normalize(),
                )
                elapsed_days = max(0, int((current_day - previous_day).days))
                swap_cost = daily_swap * elapsed_days
                last_swap_dates[position.position_id] = current_day
            else:
                swap_cost = daily_swap
            position.swap_paid += swap_cost
            cash_balance -= swap_cost

        if current_date in signal_dates:
            signal = signals.loc[current_date]
            prediction = float(signal["prediction"])
            lot_size = float(signal["lot_size"])
            confidence = float(signal.get("confidence", np.nan))
            expected_change_pct = (prediction / close - 1) * 100
            buy_count = sum(position.direction == "BUY" for position in open_positions)
            sell_count = sum(position.direction == "SELL" for position in open_positions)

            if expected_change_pct > 0 and expected_change_pct >= entry_threshold_pct:
                direction = "BUY"
                max_positions = max_buy_positions
                can_open = buy_count < max_buy_positions
            elif expected_change_pct < 0 and abs(expected_change_pct) >= entry_threshold_pct:
                direction = "SELL"
                max_positions = max_sell_positions
                can_open = sell_count < max_sell_positions
            else:
                can_open = False

            if can_open and live_rules and direction in last_cl_price:
                if direction == "SELL":
                    can_open = close <= last_cl_price[direction] - LIVE_REENTRY_BUFFER_USD
                else:
                    can_open = close >= last_cl_price[direction] + LIVE_REENTRY_BUFFER_USD

            if can_open and risk_cap_pct is not None:
                current_unrealized = sum(_dynamic_unrealized(position, close) for position in open_positions)
                current_equity = cash_balance + current_unrealized
                open_risk = sum(
                    position.stop_loss_usd if position.stop_loss_usd is not None else position.take_profit_usd
                    for position in open_positions
                )
                new_risk = stop_loss_usd if stop_loss_usd is not None else take_profit_usd
                can_open = (open_risk + new_risk) <= current_equity * (risk_cap_pct / 100)

            if can_open:
                open_positions.append(
                    DynamicPosition(
                        position_id=next_position_id,
                        phase=phase,
                        model_name=model_name,
                        strategy_name=strategy_name,
                        signal_date=current_date,
                        direction=direction,
                        lot_size=lot_size,
                        confidence=confidence,
                        entry_price=close,
                        prediction=prediction,
                        expected_change_pct=expected_change_pct,
                        take_profit_usd=take_profit_usd,
                        stop_loss_usd=stop_loss_usd,
                        profit_close_usd=profit_close_usd,
                        profit_protection_activation_usd=profit_protection_activation_usd,
                        profit_protection_floor_usd=profit_protection_floor_usd,
                        profit_protection_trail_usd=profit_protection_trail_usd,
                        entry_threshold_pct=entry_threshold_pct,
                        max_positions=max_positions,
                    )
                )
                next_position_id += 1

        unrealized = sum(_dynamic_unrealized(position, close) for position in open_positions)
        equity = cash_balance + unrealized
        equity_rows.append(
            {
                "Tanggal": current_date,
                "Fase": phase,
                "Balance": cash_balance,
                "Equity": equity,
                "Unrealized P/L": unrealized,
                "Open BUY": sum(position.direction == "BUY" for position in open_positions),
                "Open SELL": sum(position.direction == "SELL" for position in open_positions),
                "Open total": len(open_positions),
                "Target equity tercapai": equity >= target_equity,
            }
        )

        if close_on_target_equity and equity >= target_equity:
            for position in open_positions:
                cash_balance += _dynamic_unrealized(position, close)
                closed_rows.append(_dynamic_close_row(position, current_date, close, "Target equity tercapai", cash_balance))
            open_positions = []
            equity_rows[-1]["Balance"] = cash_balance
            equity_rows[-1]["Equity"] = cash_balance
            equity_rows[-1]["Unrealized P/L"] = 0.0
            equity_rows[-1]["Open BUY"] = 0
            equity_rows[-1]["Open SELL"] = 0
            equity_rows[-1]["Open total"] = 0
            break

    if open_positions:
        final_date = gold_ohlc.index[-1]
        final_close = float(gold_ohlc.iloc[-1]["Close"])
        for position in open_positions:
            cash_balance += _dynamic_unrealized(position, final_close)
            closed_rows.append(_dynamic_close_row(position, final_date, final_close, "Akhir periode data", cash_balance))
        if equity_rows and equity_rows[-1]["Tanggal"] == final_date:
            equity_rows[-1]["Balance"] = cash_balance
            equity_rows[-1]["Equity"] = cash_balance
            equity_rows[-1]["Unrealized P/L"] = 0.0
            equity_rows[-1]["Open BUY"] = 0
            equity_rows[-1]["Open SELL"] = 0
            equity_rows[-1]["Open total"] = 0

    return _result(closed_rows, equity_rows, initial_balance, target_equity)


def _phase_row(phase: int, result: SimulationResult, start_equity: float, target_equity: float) -> dict[str, object]:
    summary = result.summary
    max_open_total = 0.0
    if not result.equity_curve.empty and "Open total" in result.equity_curve.columns:
        max_open_total = float(pd.to_numeric(result.equity_curve["Open total"], errors="coerce").max())
    return {
        "Fase": phase,
        "Start equity": start_equity,
        "Target equity": target_equity,
        "Equity close-all": summary["Equity akhir"],
        "Target tercapai": summary["Target tercapai"],
        "Tanggal target": summary["Tanggal target"],
        "Equity terendah": summary["Equity terendah"],
        "Tanggal equity terendah": summary["Tanggal equity terendah"],
        "Equity tertinggi": summary["Equity tertinggi"],
        "Tanggal equity tertinggi": summary["Tanggal equity tertinggi"],
        "Total net P/L": summary["Total net P/L"],
        "Total swap": summary["Total swap"],
        "Jumlah transaksi": summary["Jumlah transaksi"],
        "Total BUY": summary["Total BUY"],
        "Total SELL": summary["Total SELL"],
        "Max open posisi": max_open_total,
        "Win rate": summary["Win rate"],
        "Max drawdown": summary["Max drawdown"],
        "Profit factor": summary["Profit factor"],
        "Status": "Selesai" if summary["Target tercapai"] else "Berjalan sampai akhir periode",
    }


def _multiphase_result(
    signals: pd.DataFrame,
    gold_ohlc: pd.DataFrame,
    model_name: str,
    *,
    strategy_name: str,
    take_profit_usd: float,
    stop_loss_usd: float,
    entry_threshold_pct: float,
    live_rules: bool = False,
    max_buy_positions: int = 8,
    max_sell_positions: int = 10,
    risk_cap_pct: float | None = None,
    phase_growth: float = PHASE_GROWTH,
    profit_close_usd: float | None = None,
    profit_protection_activation_usd: float | None = None,
    profit_protection_floor_usd: float | None = None,
    profit_protection_trail_usd: float | None = None,
    close_on_target_equity: bool = True,
    accrue_swap_by_elapsed_days: bool = False,
    test_start: pd.Timestamp = OPTIMIZATION_START,
    test_end: pd.Timestamp = OPTIMIZATION_END,
) -> MultiPhaseSimulationResult:
    clean_gold = gold_ohlc.loc[(gold_ohlc.index >= test_start) & (gold_ohlc.index <= test_end)].copy()
    clean_signals = signals.loc[(signals.index >= test_start) & (signals.index <= test_end)].copy()
    phase = 1
    start_equity = INITIAL_EQUITY
    phase_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    cursor_date = test_start - pd.Timedelta(days=1)

    while cursor_date < test_end:
        phase_signals = clean_signals[clean_signals.index > cursor_date]
        phase_gold = clean_gold[clean_gold.index > cursor_date]
        if phase_signals.empty or phase_gold.empty:
            break

        target_equity = start_equity * (1 + phase_growth)
        result = _simulate_phase(
            phase_signals,
            phase_gold,
            model_name,
            phase=phase,
            initial_balance=start_equity,
            target_equity=target_equity,
            take_profit_usd=take_profit_usd,
            stop_loss_usd=stop_loss_usd,
            max_buy_positions=max_buy_positions,
            max_sell_positions=max_sell_positions,
            entry_threshold_pct=entry_threshold_pct,
            strategy_name=strategy_name,
            live_rules=live_rules,
            risk_cap_pct=risk_cap_pct,
            profit_close_usd=profit_close_usd,
            profit_protection_activation_usd=profit_protection_activation_usd,
            profit_protection_floor_usd=profit_protection_floor_usd,
            profit_protection_trail_usd=profit_protection_trail_usd,
            close_on_target_equity=close_on_target_equity,
            accrue_swap_by_elapsed_days=accrue_swap_by_elapsed_days,
        )
        phase_rows.append(_phase_row(phase, result, start_equity, target_equity))
        if not result.trades.empty:
            trade_frames.append(result.trades)
        if not result.equity_curve.empty:
            equity_frames.append(result.equity_curve)

        if (
            not close_on_target_equity
            or not result.summary["Target tercapai"]
            or pd.isna(result.summary["Tanggal target"])
        ):
            break

        cursor_date = pd.Timestamp(result.summary["Tanggal target"])
        start_equity = float(result.summary["Equity akhir"])
        phase += 1

    phases = pd.DataFrame(phase_rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    equity_curve = pd.concat(equity_frames).sort_index() if equity_frames else pd.DataFrame()
    if equity_curve.index.name is None:
        equity_curve.index.name = "Tanggal"

    if phases.empty:
        summary = _result([], [], INITIAL_EQUITY, INITIAL_EQUITY * (1 + phase_growth)).summary
        summary.update({"Fase selesai": 0.0, "Fase total": 0.0, "Growth total": 0.0})
        return MultiPhaseSimulationResult(summary, phases, trades, equity_curve)

    final_equity = float(phases["Equity close-all"].iloc[-1])
    completed = phases[phases["Target tercapai"]]
    if trades.empty:
        net_pnl = pd.Series(dtype=float)
        total_buy = total_sell = 0.0
        win_rate = np.nan
        profit_factor = np.nan
        avg_net = 0.0
        total_swap = 0.0
    else:
        net_pnl = pd.to_numeric(trades["Net P/L"], errors="coerce")
        total_buy = float((trades["Arah"] == "BUY").sum())
        total_sell = float((trades["Arah"] == "SELL").sum())
        win_rate = float((net_pnl > 0).mean() * 100)
        gross_profit = float(net_pnl[net_pnl > 0].sum())
        gross_loss = abs(float(net_pnl[net_pnl < 0].sum()))
        profit_factor = np.nan if gross_loss == 0 else gross_profit / gross_loss
        avg_net = float(net_pnl.mean()) if not net_pnl.empty else 0.0
        total_swap = float(pd.to_numeric(trades["Swap"], errors="coerce").sum())

    if equity_curve.empty:
        lowest_equity = highest_equity = final_equity
        low_date = high_date = None
        max_drawdown = 0.0
    else:
        equity = pd.to_numeric(equity_curve["Equity"], errors="coerce")
        lowest_equity = float(equity.min())
        highest_equity = float(equity.max())
        low_date = equity.idxmin()
        high_date = equity.idxmax()
        max_drawdown = float((equity.cummax() - equity).max())

    summary = {
        "Modal awal": INITIAL_EQUITY,
        "Balance akhir": final_equity,
        "Equity akhir": final_equity,
        "Target equity": float(phases["Target equity"].iloc[-1]),
        "Target tercapai": bool(phases["Target tercapai"].iloc[-1]),
        "Tanggal target": phases["Tanggal target"].iloc[-1],
        "Fase selesai": float(len(completed)),
        "Fase total": float(len(phases)),
        "Growth total": (final_equity / INITIAL_EQUITY - 1) * 100,
        "Equity tertinggi": highest_equity,
        "Tanggal equity tertinggi": high_date,
        "Equity terendah": lowest_equity,
        "Tanggal equity terendah": low_date,
        "Total net P/L": float(net_pnl.sum()) if not net_pnl.empty else 0.0,
        "Jumlah transaksi": float(len(trades)),
        "Win rate": win_rate,
        "Max drawdown": max_drawdown,
        "Total BUY": total_buy,
        "Total SELL": total_sell,
        "Max open posisi": float(pd.to_numeric(equity_curve["Open total"], errors="coerce").max()) if not equity_curve.empty and "Open total" in equity_curve.columns else 0.0,
        "Profit factor": float(profit_factor) if not pd.isna(profit_factor) else np.nan,
        "Avg net P/L": avg_net,
        "Total swap": total_swap,
    }
    return MultiPhaseSimulationResult(summary, phases, trades, equity_curve)


def run_optimized_strategy(
    gold_ohlc: pd.DataFrame,
    *,
    phase_growth: float = PHASE_GROWTH,
    model_name: str = "Strategi Optimizer",
    profit_close_usd: float | None = None,
    profit_protection_activation_usd: float | None = None,
    profit_protection_floor_usd: float | None = None,
    profit_protection_trail_usd: float | None = None,
    close_on_target_equity: bool = True,
    test_start: pd.Timestamp = OPTIMIZATION_START,
    test_end: pd.Timestamp = OPTIMIZATION_END,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    candidates: list[dict[str, object]] = []
    modes = ["Trend", "Breakout", "Pullback"]
    fast_windows = [10, 20]
    slow_windows = [50]
    momentum_days_options = [5, 10]
    thresholds = [0.15]
    take_profits = [15.0, 25.0]
    stop_losses = [10.0, 18.0]
    lot_sizes = [0.01, 0.02]

    for mode in modes:
        for fast_window in fast_windows:
            for slow_window in slow_windows:
                if fast_window >= slow_window:
                    continue
                for momentum_days in momentum_days_options:
                    for threshold in thresholds:
                        predictions = _indicator_predictions(
                            gold_ohlc,
                            mode,
                            fast_window,
                            slow_window,
                            momentum_days,
                            threshold,
                            test_start=test_start,
                            test_end=test_end,
                        )
                        if predictions.empty:
                            continue
                        for take_profit in take_profits:
                            for stop_loss in stop_losses:
                                for lot_size in lot_sizes:
                                    strategy_name = (
                                        f"{mode} | MA {fast_window}/{slow_window} | Mom {momentum_days} | "
                                        f"TP {take_profit:g} SL {stop_loss:g} | Lot {lot_size:.2f}"
                                    )
                                    if profit_close_usd is not None:
                                        strategy_name = f"{strategy_name} | Profit close {profit_close_usd:g}"
                                    if profit_protection_activation_usd is not None:
                                        strategy_name = (
                                            f"{strategy_name} | Protection {profit_protection_activation_usd:g}/"
                                            f"{profit_protection_floor_usd:g}/{profit_protection_trail_usd:g}"
                                        )
                                    result = _multiphase_result(
                                        _fixed_lot_signals(predictions, lot_size),
                                        gold_ohlc,
                                        model_name,
                                        strategy_name=strategy_name,
                                        take_profit_usd=take_profit,
                                        stop_loss_usd=stop_loss,
                                        entry_threshold_pct=threshold,
                                        phase_growth=phase_growth,
                                        profit_close_usd=profit_close_usd,
                                        profit_protection_activation_usd=profit_protection_activation_usd,
                                        profit_protection_floor_usd=profit_protection_floor_usd,
                                        profit_protection_trail_usd=profit_protection_trail_usd,
                                        close_on_target_equity=close_on_target_equity,
                                        test_start=test_start,
                                        test_end=test_end,
                                    )
                                    summary = result.summary
                                    if summary["Jumlah transaksi"] < 3:
                                        continue
                                    candidates.append(
                                        {
                                            "Mode": mode,
                                            "Strategi": strategy_name,
                                            "Fast MA": fast_window,
                                            "Slow MA": slow_window,
                                            "Momentum hari": momentum_days,
                                            "Threshold entry (%)": threshold,
                                            "TP (USD)": take_profit,
                                            "SL (USD)": stop_loss,
                                            "Lot": lot_size,
                                            "Target fase (%)": phase_growth * 100,
                                            "Floating profit close (USD)": profit_close_usd,
                                            "Profit protection aktif (USD)": profit_protection_activation_usd,
                                            "Profit protection floor (USD)": profit_protection_floor_usd,
                                            "Profit protection trail (USD)": profit_protection_trail_usd,
                                            "Close-all target equity": close_on_target_equity,
                                            "Fase selesai": summary["Fase selesai"],
                                            "Fase total": summary["Fase total"],
                                            "Equity akhir": summary["Equity akhir"],
                                            "Growth total": summary["Growth total"],
                                            "Equity terendah": summary["Equity terendah"],
                                            "Equity tertinggi": summary["Equity tertinggi"],
                                            "Max drawdown": summary["Max drawdown"],
                                            "Jumlah transaksi": summary["Jumlah transaksi"],
                                            "Total BUY": summary["Total BUY"],
                                            "Total SELL": summary["Total SELL"],
                                            "Total swap": summary["Total swap"],
                                            "Win rate": summary["Win rate"],
                                            "Profit factor": summary["Profit factor"],
                                            "Avg net P/L": summary["Avg net P/L"],
                                            "_score": _strategy_score(summary),
                                            "_result": result,
                                        }
                                    )

    if not candidates:
        empty = _multiphase_result(
            pd.DataFrame(),
            gold_ohlc,
            model_name,
            strategy_name="-",
            take_profit_usd=0,
            stop_loss_usd=0,
            entry_threshold_pct=0,
            phase_growth=phase_growth,
            profit_close_usd=profit_close_usd,
            profit_protection_activation_usd=profit_protection_activation_usd,
            profit_protection_floor_usd=profit_protection_floor_usd,
            profit_protection_trail_usd=profit_protection_trail_usd,
            close_on_target_equity=close_on_target_equity,
            test_start=test_start,
            test_end=test_end,
        )
        return empty, pd.DataFrame()

    candidates.sort(key=lambda row: row["_score"], reverse=True)
    best_result = candidates[0]["_result"]
    leaderboard = pd.DataFrame([{key: value for key, value in row.items() if not key.startswith("_")} for row in candidates])
    return best_result, leaderboard


def run_optimized_strategy_v5(
    gold_ohlc: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    return run_optimized_strategy(
        gold_ohlc,
        phase_growth=0.30,
        model_name="Strategi Optimizer v5",
    )


def run_optimized_strategy_v6(
    gold_ohlc: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    return run_optimized_strategy(
        gold_ohlc,
        phase_growth=PHASE_GROWTH,
        model_name="Strategi Optimizer v6",
        profit_close_usd=50.0,
    )


def run_optimized_strategy_v7(
    gold_ohlc: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    return run_optimized_strategy(
        gold_ohlc,
        phase_growth=PHASE_GROWTH,
        model_name="Strategi Optimizer v7",
        profit_protection_activation_usd=50.0,
        profit_protection_floor_usd=35.0,
        profit_protection_trail_usd=15.0,
    )


def run_optimized_strategy_v8(
    gold_ohlc: pd.DataFrame,
    *,
    test_start: pd.Timestamp = OPTIMIZATION_START,
    test_end: pd.Timestamp = OPTIMIZATION_END,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    return run_optimized_strategy(
        gold_ohlc,
        phase_growth=PHASE_GROWTH,
        model_name="Strategi Optimizer v8",
        profit_protection_activation_usd=50.0,
        profit_protection_floor_usd=35.0,
        profit_protection_trail_usd=15.0,
        close_on_target_equity=False,
        test_start=test_start,
        test_end=test_end,
    )


def run_optimized_strategy_v9(
    gold_ohlc: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    protection_sets = [
        (40.0, 25.0, 10.0),
        (50.0, 35.0, 15.0),
        (60.0, 40.0, 20.0),
        (75.0, 50.0, 25.0),
    ]
    candidates: list[dict[str, object]] = []

    for activation, floor, trail in protection_sets:
        result, leaderboard = run_optimized_strategy(
            gold_ohlc,
            phase_growth=PHASE_GROWTH,
            model_name="Strategi Optimizer v9",
            profit_protection_activation_usd=activation,
            profit_protection_floor_usd=floor,
            profit_protection_trail_usd=trail,
            close_on_target_equity=False,
        )
        if leaderboard.empty:
            continue
        top_row = leaderboard.iloc[0].to_dict()
        top_row["Protection preset"] = f"{activation:g}/{floor:g}/{trail:g}"
        top_row["_score"] = _strategy_score(result.summary)
        top_row["_result"] = result
        candidates.append(top_row)

    if not candidates:
        empty = _multiphase_result(
            pd.DataFrame(),
            gold_ohlc,
            "Strategi Optimizer v9",
            strategy_name="-",
            take_profit_usd=0,
            stop_loss_usd=0,
            entry_threshold_pct=0,
            close_on_target_equity=False,
        )
        return empty, pd.DataFrame()

    candidates.sort(key=lambda row: row["_score"], reverse=True)
    best_result = candidates[0]["_result"]
    leaderboard = pd.DataFrame([{key: value for key, value in row.items() if not key.startswith("_")} for row in candidates])
    return best_result, leaderboard


def run_optimized_strategy_v10(
    gold_ohlc: pd.DataFrame,
    *,
    optimization_start: pd.Timestamp = OPTIMIZATION_START,
    optimization_end: pd.Timestamp = OPTIMIZATION_END,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    signal_configs = [
        (mode, fast_window, slow_window, momentum_days, threshold)
        for mode in ["Trend", "Breakout", "Pullback"]
        for fast_window in [5, 10, 20]
        for slow_window in [50, 100]
        for momentum_days in [5, 10, 14]
        for threshold in [0.10, 0.15, 0.25]
        if fast_window < slow_window
    ]
    risk_configs = [
        (15.0, 18.0, 0.02, 8, 10, None, 50.0, 35.0, 15.0),
        (25.0, 18.0, 0.02, 8, 10, None, 50.0, 35.0, 15.0),
        (35.0, 18.0, 0.02, 8, 10, None, 50.0, 35.0, 15.0),
        (50.0, 18.0, 0.02, 8, 10, None, 50.0, 35.0, 15.0),
        (50.0, 25.0, 0.02, 8, 10, None, 75.0, 50.0, 25.0),
        (75.0, 25.0, 0.02, 8, 10, None, 75.0, 50.0, 25.0),
        (35.0, 12.0, 0.02, 8, 10, 45.0, 50.0, 35.0, 15.0),
        (50.0, 18.0, 0.03, 8, 10, 50.0, 60.0, 40.0, 20.0),
        (75.0, 25.0, 0.03, 8, 10, 50.0, 75.0, 50.0, 25.0),
        (50.0, 25.0, 0.02, 12, 12, 50.0, 60.0, 40.0, 20.0),
        (75.0, 35.0, 0.02, 12, 12, 60.0, 100.0, 65.0, 30.0),
        (100.0, 35.0, 0.02, 12, 12, 60.0, 125.0, 80.0, 40.0),
    ]
    candidates: list[dict[str, object]] = []
    baseline_result, baseline_leaderboard = run_optimized_strategy_v8(
        gold_ohlc,
        test_start=optimization_start,
        test_end=optimization_end,
    )
    if not baseline_leaderboard.empty:
        baseline_row = baseline_leaderboard.iloc[0].to_dict()
        baseline_summary = baseline_result.summary
        baseline_row["Eksplorasi"] = "Baseline v8"
        baseline_row["_score"] = (
            float(baseline_summary["Equity akhir"]),
            -float(baseline_summary["Max drawdown"]),
            float(baseline_summary["Profit factor"]) if not pd.isna(baseline_summary["Profit factor"]) else 0.0,
            float(baseline_summary["Jumlah transaksi"]),
        )
        baseline_row["_result"] = baseline_result
        candidates.append(baseline_row)

    for mode, fast_window, slow_window, momentum_days, threshold in signal_configs:
        predictions = _indicator_predictions(
            gold_ohlc,
            mode,
            fast_window,
            slow_window,
            momentum_days,
            threshold,
            test_start=optimization_start,
            test_end=optimization_end,
        )
        if predictions.empty:
            continue
        for (
            take_profit,
            stop_loss,
            lot_size,
            max_buy,
            max_sell,
            risk_cap,
            protection_activation,
            protection_floor,
            protection_trail,
        ) in risk_configs:
            strategy_name = (
                f"{mode} | MA {fast_window}/{slow_window} | Mom {momentum_days} | "
                f"TP {take_profit:g} SL {stop_loss:g} | Lot {lot_size:.2f} | "
                f"Max {max_buy}/{max_sell} | Protection {protection_activation:g}/"
                f"{protection_floor:g}/{protection_trail:g}"
            )
            result = _multiphase_result(
                _fixed_lot_signals(predictions, lot_size),
                gold_ohlc,
                "Strategi Optimizer v10",
                strategy_name=strategy_name,
                take_profit_usd=take_profit,
                stop_loss_usd=stop_loss,
                entry_threshold_pct=threshold,
                max_buy_positions=max_buy,
                max_sell_positions=max_sell,
                risk_cap_pct=risk_cap,
                phase_growth=PHASE_GROWTH,
                profit_protection_activation_usd=protection_activation,
                profit_protection_floor_usd=protection_floor,
                profit_protection_trail_usd=protection_trail,
                close_on_target_equity=False,
                test_start=optimization_start,
                test_end=optimization_end,
            )
            summary = result.summary
            if summary["Jumlah transaksi"] < 3:
                continue
            candidates.append(
                {
                    "Eksplorasi": "Expanded fixed lot",
                    "Mode": mode,
                    "Strategi": strategy_name,
                    "Fast MA": fast_window,
                    "Slow MA": slow_window,
                    "Momentum hari": momentum_days,
                    "Threshold entry (%)": threshold,
                    "TP (USD)": take_profit,
                    "SL (USD)": stop_loss,
                    "Lot": lot_size,
                    "Max BUY": max_buy,
                    "Max SELL": max_sell,
                    "Risk cap floating SL (%)": risk_cap,
                    "Target fase (%)": PHASE_GROWTH * 100,
                    "Profit protection aktif (USD)": protection_activation,
                    "Profit protection floor (USD)": protection_floor,
                    "Profit protection trail (USD)": protection_trail,
                    "Close-all target equity": False,
                    "Fase selesai": summary["Fase selesai"],
                    "Fase total": summary["Fase total"],
                    "Equity akhir": summary["Equity akhir"],
                    "Growth total": summary["Growth total"],
                    "Equity terendah": summary["Equity terendah"],
                    "Equity tertinggi": summary["Equity tertinggi"],
                    "Max drawdown": summary["Max drawdown"],
                    "Jumlah transaksi": summary["Jumlah transaksi"],
                    "Total BUY": summary["Total BUY"],
                    "Total SELL": summary["Total SELL"],
                    "Max open posisi": summary["Max open posisi"],
                    "Total swap": summary["Total swap"],
                    "Win rate": summary["Win rate"],
                    "Profit factor": summary["Profit factor"],
                    "Avg net P/L": summary["Avg net P/L"],
                    "_score": (
                        float(summary["Equity akhir"]),
                        -float(summary["Max drawdown"]),
                        float(summary["Profit factor"]) if not pd.isna(summary["Profit factor"]) else 0.0,
                        float(summary["Jumlah transaksi"]),
                    ),
                    "_result": result,
                }
            )

    if not candidates:
        return run_optimized_strategy_v8(gold_ohlc)

    candidates.sort(key=lambda row: row["_score"], reverse=True)
    best_result = candidates[0]["_result"]
    leaderboard = pd.DataFrame([{key: value for key, value in row.items() if not key.startswith("_")} for row in candidates])
    return best_result, leaderboard


def run_optimized_strategy_v10_real_data(
    gold_ohlc: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    _, v10_leaderboard = run_optimized_strategy_v10(gold_ohlc)
    if v10_leaderboard.empty:
        empty = _multiphase_result(
            pd.DataFrame(),
            gold_ohlc,
            "Strategi Optimizer v10 Data Real",
            strategy_name="-",
            take_profit_usd=0,
            stop_loss_usd=0,
            entry_threshold_pct=0,
            close_on_target_equity=False,
            test_start=REAL_DATA_TEST_START,
            test_end=REAL_DATA_TEST_END,
        )
        return empty, pd.DataFrame()

    best = v10_leaderboard.iloc[0].to_dict()
    mode = str(best["Mode"])
    fast_window = int(best["Fast MA"])
    slow_window = int(best["Slow MA"])
    momentum_days = int(best["Momentum hari"])
    threshold = float(best["Threshold entry (%)"])
    take_profit = float(best["TP (USD)"])
    stop_loss = float(best["SL (USD)"])
    lot_size = float(best.get("Lot", 0.02))
    max_buy = int(best.get("Max BUY", 8))
    max_sell = int(best.get("Max SELL", 10))
    risk_cap = best.get("Risk cap floating SL (%)")
    risk_cap = None if pd.isna(risk_cap) else float(risk_cap)
    protection_activation = best.get("Profit protection aktif (USD)")
    protection_floor = best.get("Profit protection floor (USD)")
    protection_trail = best.get("Profit protection trail (USD)")
    protection_activation = None if pd.isna(protection_activation) else float(protection_activation)
    protection_floor = None if pd.isna(protection_floor) else float(protection_floor)
    protection_trail = None if pd.isna(protection_trail) else float(protection_trail)
    strategy_name = f"{best.get('Strategi', 'Strategi Optimizer v10')} | Real data 1-16 Jul 2026"

    predictions = _indicator_predictions(
        gold_ohlc,
        mode,
        fast_window,
        slow_window,
        momentum_days,
        threshold,
        test_start=REAL_DATA_TEST_START,
        test_end=REAL_DATA_TEST_END,
    )
    result = _multiphase_result(
        _fixed_lot_signals(predictions, lot_size),
        gold_ohlc,
        "Strategi Optimizer v10 Data Real",
        strategy_name=strategy_name,
        take_profit_usd=take_profit,
        stop_loss_usd=stop_loss,
        entry_threshold_pct=threshold,
        max_buy_positions=max_buy,
        max_sell_positions=max_sell,
        risk_cap_pct=risk_cap,
        phase_growth=PHASE_GROWTH,
        profit_protection_activation_usd=protection_activation,
        profit_protection_floor_usd=protection_floor,
        profit_protection_trail_usd=protection_trail,
        close_on_target_equity=False,
        test_start=REAL_DATA_TEST_START,
        test_end=REAL_DATA_TEST_END,
    )
    summary = result.summary
    summary["Periode uji"] = "1 Jul 2026 - 16 Jul 2026"
    summary["Sumber parameter"] = "Best candidate Optimizer v10"
    leaderboard = pd.DataFrame(
        [
            {
                "Periode uji": "1 Jul 2026 - 16 Jul 2026",
                "Sumber parameter": "Best candidate Optimizer v10",
                "Mode": mode,
                "Strategi": strategy_name,
                "Fast MA": fast_window,
                "Slow MA": slow_window,
                "Momentum hari": momentum_days,
                "Threshold entry (%)": threshold,
                "TP (USD)": take_profit,
                "SL (USD)": stop_loss,
                "Lot": lot_size,
                "Max BUY": max_buy,
                "Max SELL": max_sell,
                "Risk cap floating SL (%)": risk_cap,
                "Target fase (%)": PHASE_GROWTH * 100,
                "Profit protection aktif (USD)": protection_activation,
                "Profit protection floor (USD)": protection_floor,
                "Profit protection trail (USD)": protection_trail,
                "Close-all target equity": False,
                "Fase selesai": summary["Fase selesai"],
                "Fase total": summary["Fase total"],
                "Equity akhir": summary["Equity akhir"],
                "Growth total": summary["Growth total"],
                "Equity terendah": summary["Equity terendah"],
                "Equity tertinggi": summary["Equity tertinggi"],
                "Max drawdown": summary["Max drawdown"],
                "Jumlah transaksi": summary["Jumlah transaksi"],
                "Total BUY": summary["Total BUY"],
                "Total SELL": summary["Total SELL"],
                "Max open posisi": summary["Max open posisi"],
                "Total swap": summary["Total swap"],
                "Win rate": summary["Win rate"],
                "Profit factor": summary["Profit factor"],
                "Avg net P/L": summary["Avg net P/L"],
            }
        ]
    )
    return result, leaderboard


def _run_v10_best_on_period(
    gold_ohlc: pd.DataFrame,
    best: dict[str, object],
    *,
    model_name: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    strategy_suffix: str,
) -> MultiPhaseSimulationResult:
    mode = str(best["Mode"])
    fast_window = int(best["Fast MA"])
    slow_window = int(best["Slow MA"])
    momentum_days = int(best["Momentum hari"])
    threshold = float(best["Threshold entry (%)"])
    take_profit = float(best["TP (USD)"])
    stop_loss = float(best["SL (USD)"])
    lot_size = float(best.get("Lot", 0.02))
    max_buy = int(best.get("Max BUY", 8))
    max_sell = int(best.get("Max SELL", 10))
    risk_cap = best.get("Risk cap floating SL (%)")
    risk_cap = None if pd.isna(risk_cap) else float(risk_cap)
    protection_activation = best.get("Profit protection aktif (USD)")
    protection_floor = best.get("Profit protection floor (USD)")
    protection_trail = best.get("Profit protection trail (USD)")
    protection_activation = None if pd.isna(protection_activation) else float(protection_activation)
    protection_floor = None if pd.isna(protection_floor) else float(protection_floor)
    protection_trail = None if pd.isna(protection_trail) else float(protection_trail)
    strategy_name = f"{best.get('Strategi', 'Strategi Optimizer v10')} | {strategy_suffix}"

    predictions = _indicator_predictions(
        gold_ohlc,
        mode,
        fast_window,
        slow_window,
        momentum_days,
        threshold,
        test_start=test_start,
        test_end=test_end,
    )
    return _multiphase_result(
        _fixed_lot_signals(predictions, lot_size),
        gold_ohlc,
        model_name,
        strategy_name=strategy_name,
        take_profit_usd=take_profit,
        stop_loss_usd=stop_loss,
        entry_threshold_pct=threshold,
        max_buy_positions=max_buy,
        max_sell_positions=max_sell,
        risk_cap_pct=risk_cap,
        phase_growth=PHASE_GROWTH,
        profit_protection_activation_usd=protection_activation,
        profit_protection_floor_usd=protection_floor,
        profit_protection_trail_usd=protection_trail,
        close_on_target_equity=False,
        test_start=test_start,
        test_end=test_end,
    )


def run_optimized_strategy_v10_walk_forward(
    gold_ohlc: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    folds = [
        ("Fold 1", pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31")),
        ("Fold 2", pd.Timestamp("2024-03-31"), pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30")),
        ("Fold 3", pd.Timestamp("2024-06-30"), pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30")),
        ("Fold 4", pd.Timestamp("2024-09-30"), pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31")),
        ("Fold 5", pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31")),
        ("Fold 6", pd.Timestamp("2025-03-31"), pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30")),
        ("Fold 7", pd.Timestamp("2025-06-30"), pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30")),
        ("Fold 8", pd.Timestamp("2025-09-30"), pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31")),
        ("Fold 9", pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")),
        ("Fold 10", pd.Timestamp("2026-03-31"), pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30")),
    ]
    rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []

    for fold_name, train_end, test_start, test_end in folds:
        train_gold = gold_ohlc.loc[gold_ohlc.index <= train_end].copy()
        if train_gold.empty:
            continue
        train_result, train_leaderboard = run_optimized_strategy_v10(
            train_gold,
            optimization_start=V10_WALK_FORWARD_START,
            optimization_end=train_end,
        )
        if train_leaderboard.empty:
            continue
        best = train_leaderboard.iloc[0].to_dict()
        test_result = _run_v10_best_on_period(
            gold_ohlc,
            best,
            model_name="Strategi Optimizer v10 Walk-Forward",
            test_start=test_start,
            test_end=test_end,
            strategy_suffix=f"{fold_name}: train sampai {train_end.strftime('%d %b %Y')}",
        )
        train_summary = train_result.summary
        test_summary = test_result.summary
        test_trades = test_result.trades.copy()
        test_equity = test_result.equity_curve.copy()
        if not test_trades.empty:
            test_trades.insert(0, "Fold", fold_name)
            trade_frames.append(test_trades)
        if not test_equity.empty:
            test_equity["Fold"] = fold_name
            equity_frames.append(test_equity)

        rows.append(
            {
                "Fold": fold_name,
                "Train mulai": V10_WALK_FORWARD_START,
                "Train akhir": train_end,
                "Test mulai": test_start,
                "Test akhir": test_end,
                "Train equity akhir": train_summary["Equity akhir"],
                "Train growth (%)": train_summary["Growth total"],
                "Test equity akhir": test_summary["Equity akhir"],
                "Test growth (%)": test_summary["Growth total"],
                "Test max drawdown": test_summary["Max drawdown"],
                "Test jumlah transaksi": test_summary["Jumlah transaksi"],
                "Test total BUY": test_summary["Total BUY"],
                "Test total SELL": test_summary["Total SELL"],
                "Test win rate": test_summary["Win rate"],
                "Test profit factor": test_summary["Profit factor"],
                "Test avg net P/L": test_summary["Avg net P/L"],
                "Overfitting flag": "YA" if train_summary["Growth total"] > 20 and test_summary["Growth total"] < 0 else "Tidak",
                "Mode": best.get("Mode"),
                "Fast MA": best.get("Fast MA"),
                "Slow MA": best.get("Slow MA"),
                "Momentum hari": best.get("Momentum hari"),
                "Threshold entry (%)": best.get("Threshold entry (%)"),
                "TP (USD)": best.get("TP (USD)"),
                "SL (USD)": best.get("SL (USD)"),
                "Lot": best.get("Lot"),
                "Max BUY": best.get("Max BUY"),
                "Max SELL": best.get("Max SELL"),
                "Risk cap floating SL (%)": best.get("Risk cap floating SL (%)"),
                "Profit protection aktif (USD)": best.get("Profit protection aktif (USD)"),
                "Profit protection floor (USD)": best.get("Profit protection floor (USD)"),
                "Profit protection trail (USD)": best.get("Profit protection trail (USD)"),
                "Strategi": best.get("Strategi"),
            }
        )

    leaderboard = pd.DataFrame(rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    equity_curve = pd.concat(equity_frames).sort_index() if equity_frames else pd.DataFrame()
    if leaderboard.empty:
        summary = _result([], [], INITIAL_EQUITY, INITIAL_EQUITY * (1 + PHASE_GROWTH)).summary
        summary.update({"Fase selesai": 0.0, "Fase total": 0.0, "Growth total": 0.0})
        return MultiPhaseSimulationResult(summary, leaderboard, trades, equity_curve), leaderboard

    test_growth = pd.to_numeric(leaderboard["Test growth (%)"], errors="coerce")
    test_equity = pd.to_numeric(leaderboard["Test equity akhir"], errors="coerce")
    test_drawdown = pd.to_numeric(leaderboard["Test max drawdown"], errors="coerce")
    test_trades_count = pd.to_numeric(leaderboard["Test jumlah transaksi"], errors="coerce")
    profitable_folds = float((test_growth > 0).sum())
    overfit_flags = float((leaderboard["Overfitting flag"] == "YA").sum())
    summary = {
        "Modal awal": INITIAL_EQUITY,
        "Balance akhir": float(test_equity.mean()),
        "Equity akhir": float(test_equity.mean()),
        "Target equity": INITIAL_EQUITY * (1 + PHASE_GROWTH),
        "Target tercapai": bool((test_equity >= INITIAL_EQUITY * (1 + PHASE_GROWTH)).any()),
        "Tanggal target": pd.NaT,
        "Fase selesai": profitable_folds,
        "Fase total": float(len(leaderboard)),
        "Growth total": float(test_growth.mean()),
        "Equity tertinggi": float(test_equity.max()),
        "Tanggal equity tertinggi": pd.NaT,
        "Equity terendah": float(test_equity.min()),
        "Tanggal equity terendah": pd.NaT,
        "Total net P/L": float((test_equity - INITIAL_EQUITY).sum()),
        "Jumlah transaksi": float(test_trades_count.sum()),
        "Win rate": float((test_growth > 0).mean() * 100),
        "Max drawdown": float(test_drawdown.max()),
        "Total BUY": float(pd.to_numeric(leaderboard["Test total BUY"], errors="coerce").sum()),
        "Total SELL": float(pd.to_numeric(leaderboard["Test total SELL"], errors="coerce").sum()),
        "Max open posisi": float(pd.to_numeric(equity_curve.get("Open total", pd.Series(dtype=float)), errors="coerce").max()) if not equity_curve.empty else 0.0,
        "Profit factor": np.nan,
        "Avg net P/L": float(pd.to_numeric(leaderboard["Test avg net P/L"], errors="coerce").mean()),
        "Total swap": float(pd.to_numeric(trades.get("Swap", pd.Series(dtype=float)), errors="coerce").sum()) if not trades.empty else 0.0,
        "Fold profitable": profitable_folds,
        "Fold overfitting": overfit_flags,
        "Worst fold growth (%)": float(test_growth.min()),
        "Best fold growth (%)": float(test_growth.max()),
    }
    return MultiPhaseSimulationResult(summary, leaderboard, trades, equity_curve), leaderboard


def run_optimized_strategy_v3(
    gold_ohlc: pd.DataFrame,
    optimizer_leaderboard: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    if optimizer_leaderboard.empty:
        empty = _multiphase_result(
            pd.DataFrame(),
            gold_ohlc,
            "Strategi Optimizer v3",
            strategy_name="-",
            take_profit_usd=0,
            stop_loss_usd=0,
            entry_threshold_pct=0,
            live_rules=True,
        )
        return empty, pd.DataFrame()

    best = optimizer_leaderboard.iloc[0].to_dict()
    mode = str(best["Mode"])
    fast_window = int(best["Fast MA"])
    slow_window = int(best["Slow MA"])
    momentum_days = int(best["Momentum hari"])
    threshold = float(best["Threshold entry (%)"])
    take_profit = float(best["TP (USD)"])
    stop_loss = float(best["SL (USD)"])
    lot_size = float(best.get("Lot", 0.01))
    base_strategy = str(best.get("Strategi", "Strategi Terbaik Optimizer"))
    strategy_name = f"{base_strategy} | Live rules: anti-duplikat aktif + re-entry CL USD {LIVE_REENTRY_BUFFER_USD:g}"

    predictions = _indicator_predictions(
        gold_ohlc,
        mode,
        fast_window,
        slow_window,
        momentum_days,
        threshold,
    )
    result = _multiphase_result(
        _fixed_lot_signals(predictions, lot_size),
        gold_ohlc,
        "Strategi Optimizer v3",
        strategy_name=strategy_name,
        take_profit_usd=take_profit,
        stop_loss_usd=stop_loss,
        entry_threshold_pct=threshold,
        live_rules=True,
    )
    summary = result.summary
    leaderboard = pd.DataFrame(
        [
            {
                "Mode": mode,
                "Strategi": strategy_name,
                "Fast MA": fast_window,
                "Slow MA": slow_window,
                "Momentum hari": momentum_days,
                "Threshold entry (%)": threshold,
                "TP (USD)": take_profit,
                "SL (USD)": stop_loss,
                "Lot": lot_size,
                "Re-entry buffer (USD)": LIVE_REENTRY_BUFFER_USD,
                "Rule tambahan": "Anti-duplikat aktif, re-entry setelah CL, guard candle entry",
                "Fase selesai": summary["Fase selesai"],
                "Fase total": summary["Fase total"],
                "Equity akhir": summary["Equity akhir"],
                "Growth total": summary["Growth total"],
                "Equity terendah": summary["Equity terendah"],
                "Equity tertinggi": summary["Equity tertinggi"],
                "Max drawdown": summary["Max drawdown"],
                "Jumlah transaksi": summary["Jumlah transaksi"],
                "Total BUY": summary["Total BUY"],
                "Total SELL": summary["Total SELL"],
                "Total swap": summary["Total swap"],
                "Win rate": summary["Win rate"],
                "Profit factor": summary["Profit factor"],
                "Avg net P/L": summary["Avg net P/L"],
            }
        ]
    )
    return result, leaderboard


def run_optimized_strategy_v4(
    gold_ohlc: pd.DataFrame,
    optimizer_leaderboard: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    if optimizer_leaderboard.empty:
        empty = _multiphase_result(
            pd.DataFrame(),
            gold_ohlc,
            "Strategi Optimizer v4",
            strategy_name="-",
            take_profit_usd=0,
            stop_loss_usd=0,
            entry_threshold_pct=0,
        )
        return empty, pd.DataFrame()

    best = optimizer_leaderboard.iloc[0].to_dict()
    mode = str(best["Mode"])
    fast_window = int(best["Fast MA"])
    slow_window = int(best["Slow MA"])
    momentum_days = int(best["Momentum hari"])
    threshold = float(best["Threshold entry (%)"])
    take_profit = float(best["TP (USD)"])
    stop_loss = float(best["SL (USD)"])
    lot_size = float(best.get("Lot", 0.01))
    base_strategy = str(best.get("Strategi", "Strategi Terbaik Optimizer"))

    predictions = _indicator_predictions(
        gold_ohlc,
        mode,
        fast_window,
        slow_window,
        momentum_days,
        threshold,
    )
    signals = _fixed_lot_signals(predictions, lot_size)
    candidates: list[dict[str, object]] = []

    for risk_cap in [12.0, 18.0, 25.0, 35.0, 50.0]:
        strategy_name = (
            f"{base_strategy} | v4 dynamic risk cap {risk_cap:g}% | "
            "posisi tidak dibatasi angka tetap"
        )
        result = _multiphase_result(
            signals,
            gold_ohlc,
            "Strategi Optimizer v4",
            strategy_name=strategy_name,
            take_profit_usd=take_profit,
            stop_loss_usd=stop_loss,
            entry_threshold_pct=threshold,
            max_buy_positions=10_000,
            max_sell_positions=10_000,
            risk_cap_pct=risk_cap,
        )
        summary = result.summary
        candidates.append(
            {
                "Mode": mode,
                "Strategi": strategy_name,
                "Fast MA": fast_window,
                "Slow MA": slow_window,
                "Momentum hari": momentum_days,
                "Threshold entry (%)": threshold,
                "TP (USD)": take_profit,
                "SL (USD)": stop_loss,
                "Lot": lot_size,
                "Risk cap floating SL (%)": risk_cap,
                "Batas posisi BUY": "Tidak dibatasi angka tetap",
                "Batas posisi SELL": "Tidak dibatasi angka tetap",
                "Max open posisi": summary["Max open posisi"],
                "Fase selesai": summary["Fase selesai"],
                "Fase total": summary["Fase total"],
                "Equity akhir": summary["Equity akhir"],
                "Growth total": summary["Growth total"],
                "Equity terendah": summary["Equity terendah"],
                "Equity tertinggi": summary["Equity tertinggi"],
                "Max drawdown": summary["Max drawdown"],
                "Jumlah transaksi": summary["Jumlah transaksi"],
                "Total BUY": summary["Total BUY"],
                "Total SELL": summary["Total SELL"],
                "Total swap": summary["Total swap"],
                "Win rate": summary["Win rate"],
                "Profit factor": summary["Profit factor"],
                "Avg net P/L": summary["Avg net P/L"],
                "_score": _strategy_score(summary),
                "_result": result,
            }
        )

    candidates.sort(key=lambda row: row["_score"], reverse=True)
    best_result = candidates[0]["_result"]
    leaderboard = pd.DataFrame([{key: value for key, value in row.items() if not key.startswith("_")} for row in candidates])
    return best_result, leaderboard


def run_optimized_strategy_v2(
    gold_ohlc: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    candidates: list[dict[str, object]] = []
    modes = ["Hybrid Trend", "Volatility Breakout", "Pullback Confirm"]
    fast_windows = [10, 20]
    slow_windows = [50]
    momentum_days_options = [5, 10]
    thresholds = [0.15]
    confidence_cutoffs = [0.52, 0.62]
    take_profits = [15.0, 25.0]
    stop_losses = [10.0, 18.0]

    for mode in modes:
        for fast_window in fast_windows:
            for slow_window in slow_windows:
                if fast_window >= slow_window:
                    continue
                for momentum_days in momentum_days_options:
                    for threshold in thresholds:
                        for confidence_cutoff in confidence_cutoffs:
                            signals = _indicator_predictions_v2(
                                gold_ohlc,
                                mode,
                                fast_window,
                                slow_window,
                                momentum_days,
                                threshold,
                                confidence_cutoff,
                            )
                            if signals.empty:
                                continue
                            for take_profit in take_profits:
                                for stop_loss in stop_losses:
                                    avg_lot = float(pd.to_numeric(signals["lot_size"], errors="coerce").mean())
                                    avg_confidence = float(pd.to_numeric(signals["confidence"], errors="coerce").mean())
                                    strategy_name = (
                                        f"{mode} | MA {fast_window}/{slow_window} | Mom {momentum_days} | "
                                        f"TP {take_profit:g} SL {stop_loss:g} | Lot dinamis 0.01-0.02 | "
                                        f"Cutoff {confidence_cutoff:.0%}"
                                    )
                                    result = _multiphase_result(
                                        signals,
                                        gold_ohlc,
                                        "Strategi Terbaik v.2",
                                        strategy_name=strategy_name,
                                        take_profit_usd=take_profit,
                                        stop_loss_usd=stop_loss,
                                        entry_threshold_pct=threshold,
                                    )
                                    summary = result.summary
                                    if summary["Jumlah transaksi"] < 3:
                                        continue
                                    candidates.append(
                                        {
                                            "Mode": mode,
                                            "Strategi": strategy_name,
                                            "Fast MA": fast_window,
                                            "Slow MA": slow_window,
                                            "Momentum hari": momentum_days,
                                            "Threshold entry (%)": threshold,
                                            "Confidence cutoff": confidence_cutoff,
                                            "TP (USD)": take_profit,
                                            "SL (USD)": stop_loss,
                                            "Lot minimum": 0.01,
                                            "Lot maksimum": 0.02,
                                            "Lot rata-rata sinyal": avg_lot,
                                            "Confidence rata-rata": avg_confidence,
                                            "Fase selesai": summary["Fase selesai"],
                                            "Fase total": summary["Fase total"],
                                            "Equity akhir": summary["Equity akhir"],
                                            "Growth total": summary["Growth total"],
                                            "Equity terendah": summary["Equity terendah"],
                                            "Equity tertinggi": summary["Equity tertinggi"],
                                            "Max drawdown": summary["Max drawdown"],
                                            "Jumlah transaksi": summary["Jumlah transaksi"],
                                            "Total BUY": summary["Total BUY"],
                                            "Total SELL": summary["Total SELL"],
                                            "Total swap": summary["Total swap"],
                                            "Win rate": summary["Win rate"],
                                            "Profit factor": summary["Profit factor"],
                                            "Avg net P/L": summary["Avg net P/L"],
                                            "_score": _strategy_score(summary),
                                            "_result": result,
                                        }
                                    )

    if not candidates:
        empty = _multiphase_result(pd.DataFrame(), gold_ohlc, "Strategi Terbaik v.2", strategy_name="-", take_profit_usd=0, stop_loss_usd=0, entry_threshold_pct=0)
        return empty, pd.DataFrame()

    candidates.sort(key=lambda row: row["_score"], reverse=True)
    best_result = candidates[0]["_result"]
    leaderboard = pd.DataFrame([{key: value for key, value in row.items() if not key.startswith("_")} for row in candidates])
    return best_result, leaderboard
