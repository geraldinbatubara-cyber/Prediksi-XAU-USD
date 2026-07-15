from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_forecast.simulation import CONTRACT_OUNCES_PER_LOT, SimulationResult, _result, _simulate_predictions


OPTIMIZATION_START = pd.Timestamp("2025-01-01")
OPTIMIZATION_END = pd.Timestamp("2026-06-30")


@dataclass
class DynamicPosition:
    position_id: int
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
    entry_threshold_pct: float
    max_positions: int
    swap_paid: float = 0.0


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


def _strategy_score(summary: dict[str, float]) -> tuple[float, float, float, float]:
    target_rank = 1.0 if summary["Target tercapai"] else 0.0
    target_date = summary["Tanggal target"]
    days_to_target = 9999.0
    if target_rank and not pd.isna(target_date):
        days_to_target = float((pd.Timestamp(target_date) - OPTIMIZATION_START).days)
    return (
        target_rank,
        -days_to_target,
        float(summary["Equity akhir"]),
        -float(summary["Max drawdown"]),
    )


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
        "Threshold entry (%)": position.entry_threshold_pct,
        "Gross P/L": gross_pnl,
        "Swap": -position.swap_paid,
        "Net P/L": net_pnl,
        "Balance": cash_balance,
        "Batas posisi": position.max_positions,
    }


def _simulate_dynamic_predictions(
    signals: pd.DataFrame,
    gold_ohlc: pd.DataFrame,
    model_name: str,
    *,
    initial_balance: float = 1000.0,
    take_profit_usd: float = 15.0,
    swap_per_position: float = 0.2,
    max_buy_positions: int = 8,
    max_sell_positions: int = 10,
    entry_threshold_pct: float = 0.15,
    stop_loss_usd: float | None = 10.0,
    strategy_name: str = "Strategi Terbaik v.2",
    target_equity: float = 1200.0,
) -> SimulationResult:
    cash_balance = initial_balance
    next_position_id = 1
    closed_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    open_positions: list[DynamicPosition] = []
    if signals.empty:
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
            take_profit_points = position.take_profit_usd / units
            stop_loss_points = None if position.stop_loss_usd is None else position.stop_loss_usd / units
            if position.direction == "BUY":
                tp_price = position.entry_price + take_profit_points
                sl_price = None if stop_loss_points is None else position.entry_price - stop_loss_points
                hit_tp = high >= tp_price
                hit_sl = sl_price is not None and low <= sl_price
                if hit_sl:
                    exit_price = sl_price
                    exit_reason = "SL tersentuh"
                elif hit_tp:
                    exit_price = tp_price
                    exit_reason = "TP tersentuh"
                else:
                    still_open.append(position)
                    continue
            else:
                tp_price = position.entry_price - take_profit_points
                sl_price = None if stop_loss_points is None else position.entry_price + stop_loss_points
                hit_tp = low <= tp_price
                hit_sl = sl_price is not None and high >= sl_price
                if hit_sl:
                    exit_price = sl_price
                    exit_reason = "SL tersentuh"
                elif hit_tp:
                    exit_price = tp_price
                    exit_reason = "TP tersentuh"
                else:
                    still_open.append(position)
                    continue

            cash_balance += _dynamic_unrealized(position, exit_price)
            closed_rows.append(_dynamic_close_row(position, current_date, exit_price, exit_reason, cash_balance))
        open_positions = still_open

        for position in open_positions:
            position.swap_paid += swap_per_position
            cash_balance -= swap_per_position

        if current_date in signal_dates:
            signal = signals.loc[current_date]
            prediction = float(signal["prediction"])
            lot_size = float(signal["lot_size"])
            confidence = float(signal["confidence"])
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

            if can_open:
                open_positions.append(
                    DynamicPosition(
                        position_id=next_position_id,
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
                "Balance": cash_balance,
                "Equity": equity,
                "Unrealized P/L": unrealized,
                "Open BUY": sum(position.direction == "BUY" for position in open_positions),
                "Open SELL": sum(position.direction == "SELL" for position in open_positions),
                "Target equity tercapai": equity >= target_equity,
            }
        )

        if equity >= target_equity:
            for position in open_positions:
                cash_balance += _dynamic_unrealized(position, close)
                closed_rows.append(
                    _dynamic_close_row(position, current_date, close, "Target equity tercapai", cash_balance)
                )
            open_positions = []
            equity_rows[-1]["Balance"] = cash_balance
            equity_rows[-1]["Equity"] = cash_balance
            equity_rows[-1]["Unrealized P/L"] = 0.0
            equity_rows[-1]["Open BUY"] = 0
            equity_rows[-1]["Open SELL"] = 0
            break

    if open_positions:
        final_date = gold_ohlc.index[-1]
        final_close = float(gold_ohlc.iloc[-1]["Close"])
        for position in open_positions:
            cash_balance += _dynamic_unrealized(position, final_close)
            closed_rows.append(
                _dynamic_close_row(position, final_date, final_close, "Akhir periode data", cash_balance)
            )
        if equity_rows and equity_rows[-1]["Tanggal"] == final_date:
            equity_rows[-1]["Balance"] = cash_balance
            equity_rows[-1]["Equity"] = cash_balance
            equity_rows[-1]["Unrealized P/L"] = 0.0
            equity_rows[-1]["Open BUY"] = 0
            equity_rows[-1]["Open SELL"] = 0

    return _result(closed_rows, equity_rows, initial_balance, target_equity)


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


def run_optimized_strategy(
    gold_ohlc: pd.DataFrame,
    target_equity: float = 1200.0,
) -> tuple[SimulationResult, pd.DataFrame]:
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
                        )
                        if predictions.empty:
                            continue
                        for take_profit in take_profits:
                            for stop_loss in stop_losses:
                                for lot_size in lot_sizes:
                                    strategy_name = (
                                        f"{mode} | MA {fast_window}/{slow_window} | "
                                        f"Mom {momentum_days} | TP {take_profit:g} SL {stop_loss:g} | "
                                        f"Lot {lot_size:.2f}"
                                    )
                                    result = _simulate_predictions(
                                        predictions,
                                        gold_ohlc,
                                        "Strategi Optimizer",
                                        lot_size=lot_size,
                                        take_profit_usd=take_profit,
                                        stop_loss_usd=stop_loss,
                                        entry_threshold_pct=threshold,
                                        strategy_name=strategy_name,
                                        target_equity=target_equity,
                                    )
                                    summary = result.summary
                                    trade_count = summary["Jumlah transaksi"]
                                    if trade_count < 3:
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
                                            "Target tercapai": summary["Target tercapai"],
                                            "Tanggal target": summary["Tanggal target"],
                                            "Equity akhir": summary["Equity akhir"],
                                            "Equity terendah": summary["Equity terendah"],
                                            "Equity tertinggi": summary["Equity tertinggi"],
                                            "Max drawdown": summary["Max drawdown"],
                                            "Jumlah transaksi": trade_count,
                                            "Win rate": summary["Win rate"],
                                            "Profit factor": summary["Profit factor"],
                                            "Avg net P/L": summary["Avg net P/L"],
                                            "_score": _strategy_score(summary),
                                            "_result": result,
                                        }
                                    )

    if not candidates:
        empty_result = _result([], [], 1000.0, target_equity)
        return empty_result, pd.DataFrame()

    candidates.sort(key=lambda row: row["_score"], reverse=True)
    best_result = candidates[0]["_result"]
    leaderboard = pd.DataFrame(
        [{key: value for key, value in row.items() if not key.startswith("_")} for row in candidates]
    )
    return best_result, leaderboard


def run_optimized_strategy_v2(
    gold_ohlc: pd.DataFrame,
    target_equity: float = 1200.0,
) -> tuple[SimulationResult, pd.DataFrame]:
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
                                    avg_confidence = float(
                                        pd.to_numeric(signals["confidence"], errors="coerce").mean()
                                    )
                                    strategy_name = (
                                        f"{mode} | MA {fast_window}/{slow_window} | Mom {momentum_days} | "
                                        f"TP {take_profit:g} SL {stop_loss:g} | Lot dinamis 0.01-0.02 | "
                                        f"Cutoff {confidence_cutoff:.0%}"
                                    )
                                    result = _simulate_dynamic_predictions(
                                        signals,
                                        gold_ohlc,
                                        "Strategi Terbaik v.2",
                                        take_profit_usd=take_profit,
                                        stop_loss_usd=stop_loss,
                                        entry_threshold_pct=threshold,
                                        strategy_name=strategy_name,
                                        target_equity=target_equity,
                                    )
                                    summary = result.summary
                                    trade_count = summary["Jumlah transaksi"]
                                    if trade_count < 3:
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
                                            "Target tercapai": summary["Target tercapai"],
                                            "Tanggal target": summary["Tanggal target"],
                                            "Equity akhir": summary["Equity akhir"],
                                            "Equity terendah": summary["Equity terendah"],
                                            "Equity tertinggi": summary["Equity tertinggi"],
                                            "Max drawdown": summary["Max drawdown"],
                                            "Jumlah transaksi": trade_count,
                                            "Win rate": summary["Win rate"],
                                            "Profit factor": summary["Profit factor"],
                                            "Avg net P/L": summary["Avg net P/L"],
                                            "_score": _strategy_score(summary),
                                            "_result": result,
                                        }
                                    )

    if not candidates:
        empty_result = _result([], [], 1000.0, target_equity)
        return empty_result, pd.DataFrame()

    candidates.sort(key=lambda row: row["_score"], reverse=True)
    best_result = candidates[0]["_result"]
    leaderboard = pd.DataFrame(
        [{key: value for key, value in row.items() if not key.startswith("_")} for row in candidates]
    )
    return best_result, leaderboard
