from __future__ import annotations

import pandas as pd

from gold_forecast.strategy_optimizer import (
    MultiPhaseSimulationResult,
    _fixed_lot_signals,
    _indicator_predictions,
    _multiphase_result,
)


def run_fixed_m1_strategy(
    gold_m1: pd.DataFrame,
    params: dict[str, object],
    *,
    model_name: str,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    if gold_m1.empty:
        raise ValueError("Dataset candle M1 kosong.")

    data = gold_m1.sort_index().copy()
    actual_start = max(pd.Timestamp(data.index.min()), requested_start)
    actual_end = min(pd.Timestamp(data.index.max()), requested_end)
    data = data.loc[(data.index >= actual_start) & (data.index <= actual_end)]
    if data.empty:
        raise ValueError("Tidak ada candle M1 dalam periode pengujian.")

    mode = str(params["Mode"])
    fast_window = int(params["Fast MA"])
    slow_window = int(params["Slow MA"])
    momentum_bars = int(params["Momentum hari"])
    threshold = float(params["Threshold entry (%)"])
    lot = float(params["Lot"])
    predictions = _indicator_predictions(
        data,
        mode,
        fast_window,
        slow_window,
        momentum_bars,
        threshold,
        test_start=actual_start,
        test_end=actual_end,
    )

    result = _multiphase_result(
        _fixed_lot_signals(predictions, lot),
        data,
        model_name,
        strategy_name=str(params["Strategi"]).replace("Mom ", "Mom M1 "),
        take_profit_usd=float(params["TP (USD)"]),
        stop_loss_usd=float(params["SL (USD)"]),
        entry_threshold_pct=threshold,
        max_buy_positions=int(params.get("Max BUY", 8)),
        max_sell_positions=int(params.get("Max SELL", 10)),
        risk_cap_pct=_optional_float(params.get("Risk cap floating SL (%)")),
        phase_growth=float(params.get("Target fase (%)", 20.0)) / 100,
        profit_close_usd=_optional_float(params.get("Floating profit close (USD)")),
        profit_protection_activation_usd=_optional_float(params.get("Profit protection aktif (USD)")),
        profit_protection_floor_usd=_optional_float(params.get("Profit protection floor (USD)")),
        profit_protection_trail_usd=_optional_float(params.get("Profit protection trail (USD)")),
        close_on_target_equity=bool(params.get("Close-all target equity", True)),
        accrue_swap_by_elapsed_days=True,
        test_start=actual_start,
        test_end=actual_end,
    )
    result.summary.update(
        {
            "Periode uji": f"{actual_start:%d %b %Y %H:%M} - {actual_end:%d %b %Y %H:%M}",
            "Timeframe": "M1",
            "Jumlah candle": float(len(data)),
            "Cakupan lengkap": actual_start <= requested_start and actual_end >= requested_end,
            "Periode diminta": f"{requested_start:%d %b %Y} - {requested_end:%d %b %Y}",
        }
    )

    leaderboard_row = {
        **params,
        "Strategi": str(params["Strategi"]).replace("Mom ", "Mom M1 "),
        "Timeframe": "M1",
        "Momentum candle M1": momentum_bars,
        "Periode uji aktual": result.summary["Periode uji"],
        "Jumlah candle": float(len(data)),
        "Equity akhir": result.summary["Equity akhir"],
        "Growth total": result.summary["Growth total"],
        "Max drawdown": result.summary["Max drawdown"],
        "Jumlah transaksi": result.summary["Jumlah transaksi"],
        "Total BUY": result.summary["Total BUY"],
        "Total SELL": result.summary["Total SELL"],
        "Total swap": result.summary["Total swap"],
        "Win rate": result.summary["Win rate"],
        "Profit factor": result.summary["Profit factor"],
    }
    return result, pd.DataFrame([leaderboard_row])


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
