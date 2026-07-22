from __future__ import annotations

import pandas as pd

from gold_forecast.strategy_optimizer import (
    MultiPhaseSimulationResult,
    PHASE_GROWTH,
    _fixed_lot_signals,
    _indicator_predictions,
    _multiphase_result,
    run_optimized_strategy,
)


TRAIN_START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2025-12-31")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-06-30")


def run_optimizer_oos(gold_ohlc: pd.DataFrame) -> dict[str, tuple[MultiPhaseSimulationResult, pd.DataFrame, MultiPhaseSimulationResult]]:
    v1_train, v1_leaderboard = run_optimized_strategy(
        gold_ohlc,
        test_start=TRAIN_START,
        test_end=TRAIN_END,
    )
    if v1_leaderboard.empty:
        raise ValueError("Optimizer v1 tidak menghasilkan kandidat pada periode train 2025.")
    v1_best = v1_leaderboard.iloc[0].to_dict()
    v1_oos = _run_v1_best_on_period(gold_ohlc, v1_best)
    _attach_oos_metadata(v1_train, v1_oos, v1_best)

    return {"v1": (v1_train, v1_leaderboard, v1_oos)}


def _run_v1_best_on_period(
    gold_ohlc: pd.DataFrame,
    best: dict[str, object],
) -> MultiPhaseSimulationResult:
    mode = str(best["Mode"])
    fast_window = int(best["Fast MA"])
    slow_window = int(best["Slow MA"])
    momentum_days = int(best["Momentum hari"])
    threshold = float(best["Threshold entry (%)"])
    take_profit = float(best["TP (USD)"])
    stop_loss = float(best["SL (USD)"])
    lot_size = float(best["Lot"])
    phase_growth = float(best.get("Target fase (%)", PHASE_GROWTH * 100)) / 100
    close_on_target = bool(best.get("Close-all target equity", True))
    profit_close = _optional_float(best.get("Floating profit close (USD)"))
    protection_activation = _optional_float(best.get("Profit protection aktif (USD)"))
    protection_floor = _optional_float(best.get("Profit protection floor (USD)"))
    protection_trail = _optional_float(best.get("Profit protection trail (USD)"))
    predictions = _indicator_predictions(
        gold_ohlc,
        mode,
        fast_window,
        slow_window,
        momentum_days,
        threshold,
        test_start=OOS_START,
        test_end=OOS_END,
    )
    return _multiphase_result(
        _fixed_lot_signals(predictions, lot_size),
        gold_ohlc,
        "Optimizer v1 OOS",
        strategy_name=f"{best['Strategi']} | parameter train 2025 dibekukan",
        take_profit_usd=take_profit,
        stop_loss_usd=stop_loss,
        entry_threshold_pct=threshold,
        phase_growth=phase_growth,
        profit_close_usd=profit_close,
        profit_protection_activation_usd=protection_activation,
        profit_protection_floor_usd=protection_floor,
        profit_protection_trail_usd=protection_trail,
        close_on_target_equity=close_on_target,
        test_start=OOS_START,
        test_end=OOS_END,
    )


def _attach_oos_metadata(
    train_result: MultiPhaseSimulationResult,
    oos_result: MultiPhaseSimulationResult,
    best: dict[str, object],
) -> None:
    oos_result.summary.update(
        {
            "Periode train": "01 Jan 2025 - 31 Des 2025",
            "Periode OOS": "01 Jan 2026 - 30 Jun 2026",
            "Train equity akhir": train_result.summary["Equity akhir"],
            "Train growth (%)": train_result.summary["Growth total"],
            "Train max drawdown": train_result.summary["Max drawdown"],
            "Strategi terpilih": best.get("Strategi", "-"),
            "Parameter dibekukan": True,
        }
    )


def _optional_float(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)
