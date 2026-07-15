from __future__ import annotations

import numpy as np
import pandas as pd

from gold_forecast.simulation import SimulationResult, _result, _simulate_predictions


OPTIMIZATION_START = pd.Timestamp("2025-01-01")
OPTIMIZATION_END = pd.Timestamp("2026-06-30")


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
