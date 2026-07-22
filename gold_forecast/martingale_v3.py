from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gold_forecast.martingale import TEST_END, TEST_START, _prepare_data
from gold_forecast.martingale_v2 import AdaptiveParameters, _simulate
from gold_forecast.strategy_optimizer import MultiPhaseSimulationResult, _indicator_predictions


TRAIN_END = pd.Timestamp("2025-12-31")
OOS_START = pd.Timestamp("2026-01-01")
V10_FAST_MA = 20
V10_SLOW_MA = 50
V10_MOMENTUM_DAYS = 14
V10_THRESHOLD_PCT = 0.10


def run_martingale_v3(
    gold_ohlc: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame, MultiPhaseSimulationResult, MultiPhaseSimulationResult]:
    data = _prepare_data(gold_ohlc)
    indicators = _build_optimizer_recovery_indicators(gold_ohlc, data.index)
    train_data = data.loc[data.index <= TRAIN_END]
    train_indicators = indicators.reindex(train_data.index)
    candidates: list[dict[str, object]] = []

    for max_positions in (2, 3):
        for lot_multiplier in (1.0, 1.25, 1.5):
            for spacing_atr in (0.10, 0.20, 0.35):
                for basket_risk_pct in (0.5, 1.0, 1.5):
                    for target_profit_usd in (50.0, 75.0, 100.0):
                        for max_holding_days in (10, 20):
                            params = AdaptiveParameters(
                                max_positions=max_positions,
                                lot_multiplier=lot_multiplier,
                                spacing_atr=spacing_atr,
                                basket_risk_pct=basket_risk_pct,
                                target_profit_usd=target_profit_usd,
                                max_holding_days=max_holding_days,
                                minimum_entry_margin_pct=200.0,
                                direction_mode="BOTH",
                                maximum_atr_pct=2.0,
                            )
                            result = _simulate(train_data, train_indicators, params, collect_details=False)
                            candidates.append(_candidate_row(params, result))

    candidates.sort(key=lambda row: row["_score"], reverse=True)
    base_params = candidates[0]["_params"]
    filtered: list[dict[str, object]] = []
    for direction_mode in ("BOTH", "BUY_ONLY"):
        for maximum_atr_pct in (1.5, 2.0, 3.0, np.inf):
            params = replace(
                base_params,
                direction_mode=direction_mode,
                maximum_atr_pct=maximum_atr_pct,
            )
            filtered.append(_candidate_row(params, _simulate(train_data, train_indicators, params, collect_details=False)))

    filtered.sort(key=lambda row: row["_score"], reverse=True)
    selected = filtered[0]
    params = selected["_params"]
    train_result = _simulate(train_data, train_indicators, params, collect_details=True)
    oos_data = data.loc[data.index >= OOS_START]
    oos_result = _simulate(oos_data, indicators.reindex(oos_data.index), params, collect_details=True)
    full_result = _simulate(data, indicators, params, collect_details=True)
    for result in (train_result, oos_result, full_result):
        if not result.trades.empty:
            result.trades.loc[:, "Model"] = "Martingale v3"
            result.trades.loc[:, "Strategi"] = "Optimizer v10 + recovery terkonfirmasi"

    oos_summary = oos_result.summary
    oos_drawdown_pct = float(oos_summary["Max drawdown"]) / float(oos_summary["Modal awal"]) * 100
    oos_pass = (
        oos_summary["Growth total"] > 0
        and oos_drawdown_pct <= 10.0
        and oos_summary["Stop-out basket"] == 0
        and oos_summary["Jumlah basket"] >= 5
    )
    full_result.summary.update(
        {
            "Status kelayakan": "LAYAK KANDIDAT PAPER TEST" if oos_pass else "BELUM LAYAK",
            "Sumber sinyal": "Optimizer v10 Trend | MA 20/50 | Momentum 14 | threshold 0.10%",
            "Aturan recovery": "Tambah posisi hanya setelah candle sebelumnya mengonfirmasi reversal searah",
            "Periode train": f"{train_data.index.min():%d %b %Y} - {train_data.index.max():%d %b %Y}",
            "Periode OOS": f"{oos_data.index.min():%d %b %Y} - {oos_data.index.max():%d %b %Y}",
            "OOS equity akhir": oos_summary["Equity akhir"],
            "OOS growth (%)": oos_summary["Growth total"],
            "OOS max drawdown": oos_summary["Max drawdown"],
            "OOS max drawdown (%)": oos_drawdown_pct,
            "OOS jumlah basket": oos_summary["Jumlah basket"],
            "OOS total swap": oos_summary["Total swap"],
            "OOS lolos": oos_pass,
        }
    )
    leaderboard = pd.DataFrame(
        [{key: value for key, value in row.items() if not key.startswith("_")} for row in filtered]
    )
    return full_result, leaderboard, train_result, oos_result


def _build_optimizer_recovery_indicators(gold_ohlc: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    close = gold_ohlc["Close"].astype(float)
    open_price = gold_ohlc["Open"].astype(float)
    fast = close.rolling(V10_FAST_MA).mean()
    slow = close.rolling(V10_SLOW_MA).mean()
    momentum = close.pct_change(V10_MOMENTUM_DAYS) * 100
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            gold_ohlc["High"].astype(float) - gold_ohlc["Low"].astype(float),
            (gold_ohlc["High"].astype(float) - previous_close).abs(),
            (gold_ohlc["Low"].astype(float) - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14).mean()
    regime = pd.Series("WAIT", index=gold_ohlc.index, dtype=object)
    regime.loc[(close > fast) & (fast > slow) & (momentum > V10_THRESHOLD_PCT)] = "BUY"
    regime.loc[(close < fast) & (fast < slow) & (momentum < -V10_THRESHOLD_PCT)] = "SELL"

    predictions = _indicator_predictions(
        gold_ohlc,
        "Trend",
        V10_FAST_MA,
        V10_SLOW_MA,
        V10_MOMENTUM_DAYS,
        V10_THRESHOLD_PCT,
        test_start=TEST_START,
        test_end=TEST_END,
    )
    entry_signal = pd.Series("WAIT", index=gold_ohlc.index, dtype=object)
    expected_change = pd.Series(0.0, index=gold_ohlc.index)
    common = predictions.index.intersection(gold_ohlc.index)
    expected_change.loc[common] = (predictions.loc[common] / close.loc[common] - 1) * 100
    entry_signal.loc[expected_change > 0] = "BUY"
    entry_signal.loc[expected_change < 0] = "SELL"

    confirmation = pd.Series("WAIT", index=gold_ohlc.index, dtype=object)
    confirmation.loc[(regime == "BUY") & (close > open_price)] = "BUY"
    confirmation.loc[(regime == "SELL") & (close < open_price)] = "SELL"
    recovery_permission = confirmation.shift(1).fillna("WAIT")
    return pd.DataFrame(
        {
            "ATR": atr,
            "ATR pct": atr / close * 100,
            "Regime": regime,
            "Previous regime": recovery_permission,
            "Entry signal": entry_signal,
            "Fresh entry": entry_signal,
            "Expected change (%)": expected_change,
        }
    ).reindex(index)


def _candidate_row(params: AdaptiveParameters, result: MultiPhaseSimulationResult) -> dict[str, object]:
    summary = result.summary
    drawdown_pct = float(summary["Max drawdown"]) / float(summary["Modal awal"]) * 100
    growth = float(summary["Growth total"])
    eligible = (
        growth > 0
        and drawdown_pct <= 10.0
        and summary["Stop-out basket"] == 0
        and summary["Jumlah basket"] >= 10
        and summary["Max open posisi"] >= 2
    )
    risk_adjusted = growth / max(drawdown_pct, 0.10)
    return {
        "Strategi": "Optimizer v10 + recovery terkonfirmasi",
        "Arah diizinkan": params.direction_mode,
        "Maks ATR/Close (%)": params.maximum_atr_pct,
        "Max posisi": params.max_positions,
        "Lot multiplier": params.lot_multiplier,
        "Jarak entry (ATR)": params.spacing_atr,
        "Hard basket loss (%)": params.basket_risk_pct,
        "Target basket (USD)": params.target_profit_usd,
        "Time stop (hari)": params.max_holding_days,
        "Minimum margin entry (%)": params.minimum_entry_margin_pct,
        "Train equity akhir": summary["Equity akhir"],
        "Train growth (%)": growth,
        "Train max drawdown": summary["Max drawdown"],
        "Train max drawdown (%)": drawdown_pct,
        "Train basket": summary["Jumlah basket"],
        "Train stop-out": summary["Stop-out basket"],
        "Train total swap": summary["Total swap"],
        "Train risk-adjusted score": risk_adjusted,
        "Lolos seleksi train": eligible,
        "_params": params,
        "_score": (1 if eligible else 0, risk_adjusted, growth, -drawdown_pct),
    }
