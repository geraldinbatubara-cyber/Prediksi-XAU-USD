from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_forecast.model import RidgeRegressor, _features
from gold_forecast.model_v2 import _estimator, _market_features


CONTRACT_OUNCES_PER_LOT = 100

DEFAULT_SCENARIOS = [
    {
        "Strategi": "Agresif",
        "entry_threshold_pct": 0.0,
        "take_profit_usd": 5.0,
        "stop_loss_usd": 15.0,
    },
    {
        "Strategi": "Moderate",
        "entry_threshold_pct": 0.20,
        "take_profit_usd": 7.0,
        "stop_loss_usd": 14.0,
    },
    {
        "Strategi": "Konservatif",
        "entry_threshold_pct": 0.35,
        "take_profit_usd": 10.0,
        "stop_loss_usd": 15.0,
    },
    {
        "Strategi": "TP Lebar",
        "entry_threshold_pct": 0.20,
        "take_profit_usd": 15.0,
        "stop_loss_usd": 20.0,
    },
]


@dataclass
class SimulationResult:
    summary: dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


def _trade_rows(
    predictions: pd.Series,
    gold_ohlc: pd.DataFrame,
    model_name: str,
    initial_balance: float,
    lot_size: float,
    take_profit_usd: float,
    swap_per_position: float,
    max_buy_positions: int,
    max_sell_positions: int,
    entry_threshold_pct: float = 0.0,
    stop_loss_usd: float | None = None,
    strategy_name: str = "Agresif",
) -> list[dict[str, object]]:
    units = lot_size * CONTRACT_OUNCES_PER_LOT
    take_profit_points = take_profit_usd / units
    stop_loss_points = None if stop_loss_usd is None else stop_loss_usd / units
    balance = initial_balance
    rows: list[dict[str, object]] = []

    for signal_date, prediction in predictions.items():
        if signal_date not in gold_ohlc.index:
            continue

        entry_position = gold_ohlc.index.get_loc(signal_date)
        if not isinstance(entry_position, int) or entry_position + 1 >= len(gold_ohlc):
            continue

        next_date = gold_ohlc.index[entry_position + 1]
        entry_price = float(gold_ohlc.loc[signal_date, "Close"])
        high_next = float(gold_ohlc.loc[next_date, "High"])
        low_next = float(gold_ohlc.loc[next_date, "Low"])
        close_next = float(gold_ohlc.loc[next_date, "Close"])
        expected_change_pct = (float(prediction) / entry_price - 1) * 100

        if expected_change_pct > 0 and expected_change_pct >= entry_threshold_pct:
            direction = "BUY"
            max_positions = max_buy_positions
            tp_price = entry_price + take_profit_points
            sl_price = None if stop_loss_points is None else entry_price - stop_loss_points
            hit_tp = high_next >= tp_price
            hit_sl = sl_price is not None and low_next <= sl_price
            if hit_sl:
                exit_price = sl_price
                gross_pnl = -float(stop_loss_usd)
                exit_reason = "SL tersentuh"
            elif hit_tp:
                exit_price = tp_price
                gross_pnl = take_profit_usd
                exit_reason = "TP tersentuh"
            else:
                exit_price = close_next
                gross_pnl = (exit_price - entry_price) * units
                exit_reason = "Tutup keesokan hari"
        elif expected_change_pct < 0 and abs(expected_change_pct) >= entry_threshold_pct:
            direction = "SELL"
            max_positions = max_sell_positions
            tp_price = entry_price - take_profit_points
            sl_price = None if stop_loss_points is None else entry_price + stop_loss_points
            hit_tp = low_next <= tp_price
            hit_sl = sl_price is not None and high_next >= sl_price
            if hit_sl:
                exit_price = sl_price
                gross_pnl = -float(stop_loss_usd)
                exit_reason = "SL tersentuh"
            elif hit_tp:
                exit_price = tp_price
                gross_pnl = take_profit_usd
                exit_reason = "TP tersentuh"
            else:
                exit_price = close_next
                gross_pnl = (entry_price - exit_price) * units
                exit_reason = "Tutup keesokan hari"
        else:
            continue

        if max_positions < 1:
            continue

        net_pnl = gross_pnl - swap_per_position
        balance += net_pnl
        rows.append(
            {
                "Model": model_name,
                "Strategi": strategy_name,
                "Tanggal sinyal": signal_date,
                "Waktu sinyal": "23:59 WIT",
                "Tanggal entry": signal_date,
                "Waktu entry": "23:59 WIT",
                "Tanggal tutup": next_date,
                "Waktu tutup": "Saat TP tersentuh" if exit_reason == "TP tersentuh" else "Close harian",
                "Arah": direction,
                "Lot": lot_size,
                "Prediksi": float(prediction),
                "Expected change (%)": expected_change_pct,
                "Entry": entry_price,
                "Exit": exit_price,
                "Alasan exit": exit_reason,
                "TP (USD)": take_profit_usd,
                "SL (USD)": np.nan if stop_loss_usd is None else stop_loss_usd,
                "Threshold entry (%)": entry_threshold_pct,
                "Gross P/L": gross_pnl,
                "Swap": -swap_per_position,
                "Net P/L": net_pnl,
                "Balance": balance,
                "Batas posisi": max_positions,
            }
        )

    return rows


def _summary(trades: pd.DataFrame, initial_balance: float) -> dict[str, float]:
    if trades.empty:
        return {
            "Modal awal": initial_balance,
            "Balance akhir": initial_balance,
            "Total net P/L": 0.0,
            "Jumlah transaksi": 0.0,
            "Win rate": np.nan,
            "Max drawdown": 0.0,
            "Total BUY": 0.0,
            "Total SELL": 0.0,
            "Profit factor": np.nan,
            "Avg net P/L": 0.0,
        }

    balance = pd.to_numeric(trades["Balance"], errors="coerce")
    peak = balance.cummax()
    drawdown = peak - balance
    net_pnl = pd.to_numeric(trades["Net P/L"], errors="coerce")
    gross_profit = float(net_pnl[net_pnl > 0].sum())
    gross_loss = abs(float(net_pnl[net_pnl < 0].sum()))
    profit_factor = np.nan if gross_loss == 0 else gross_profit / gross_loss
    return {
        "Modal awal": initial_balance,
        "Balance akhir": float(balance.iloc[-1]),
        "Total net P/L": float(net_pnl.sum()),
        "Jumlah transaksi": float(len(trades)),
        "Win rate": float((net_pnl > 0).mean() * 100),
        "Max drawdown": float(drawdown.max()),
        "Total BUY": float((trades["Arah"] == "BUY").sum()),
        "Total SELL": float((trades["Arah"] == "SELL").sum()),
        "Profit factor": float(profit_factor) if not pd.isna(profit_factor) else np.nan,
        "Avg net P/L": float(net_pnl.mean()),
    }


def _result(rows: list[dict[str, object]], initial_balance: float) -> SimulationResult:
    trades = pd.DataFrame(rows)
    if trades.empty:
        equity_curve = pd.DataFrame(columns=["Tanggal", "Balance"]).set_index("Tanggal")
    else:
        equity_curve = trades[["Tanggal tutup", "Balance"]].rename(columns={"Tanggal tutup": "Tanggal"})
        equity_curve = equity_curve.set_index("Tanggal")
    return SimulationResult(summary=_summary(trades, initial_balance), trades=trades, equity_curve=equity_curve)


def _model_1_predictions(market: pd.DataFrame) -> pd.Series:
    close = market["gold"].dropna()
    features = _features(close)
    dataset = features.copy()
    dataset["target"] = close.shift(-1)
    dataset = dataset.dropna()
    if len(dataset) < 250:
        raise ValueError("Minimal 250 observasi bersih diperlukan untuk simulasi Model 1.")

    split = int(len(dataset) * 0.8)
    train, test = dataset.iloc[:split], dataset.iloc[split:]
    feature_names = list(features.columns)
    estimator = RidgeRegressor(alpha=10.0)
    estimator.fit(train[feature_names], train["target"])
    return pd.Series(estimator.predict(test[feature_names]), index=test.index)


def _model_2_predictions(market: pd.DataFrame) -> pd.Series:
    features = _market_features(market)
    gold = market["gold"]
    clean_features = features.dropna()
    if len(clean_features) < 500:
        raise ValueError("Minimal 500 observasi lintas pasar diperlukan untuk simulasi Model 2.")

    dataset = clean_features.copy()
    dataset["target_return"] = gold.shift(-1) / gold - 1
    dataset = dataset.dropna()
    split = int(len(dataset) * 0.8)
    train, test = dataset.iloc[:split], dataset.iloc[split:]
    feature_names = list(clean_features.columns)
    estimator = _estimator()
    estimator.fit(train[feature_names], train["target_return"])
    predicted_return = pd.Series(estimator.predict(test[feature_names]), index=test.index)
    current = gold.reindex(test.index)
    return current * (1 + predicted_return)


def _simulate_predictions(
    predictions: pd.Series,
    gold_ohlc: pd.DataFrame,
    model_name: str,
    initial_balance: float = 1000.0,
    lot_size: float = 0.01,
    take_profit_usd: float = 5.0,
    swap_per_position: float = 0.2,
    max_buy_positions: int = 8,
    max_sell_positions: int = 10,
    entry_threshold_pct: float = 0.0,
    stop_loss_usd: float | None = None,
    strategy_name: str = "Agresif",
) -> SimulationResult:
    rows = _trade_rows(
        predictions,
        gold_ohlc,
        model_name,
        initial_balance,
        lot_size,
        take_profit_usd,
        swap_per_position,
        max_buy_positions,
        max_sell_positions,
        entry_threshold_pct,
        stop_loss_usd,
        strategy_name,
    )
    return _result(rows, initial_balance)


def simulate_model_1(
    market: pd.DataFrame,
    gold_ohlc: pd.DataFrame,
    initial_balance: float = 1000.0,
    lot_size: float = 0.01,
    take_profit_usd: float = 5.0,
    swap_per_position: float = 0.2,
    max_buy_positions: int = 8,
    max_sell_positions: int = 10,
    entry_threshold_pct: float = 0.0,
    stop_loss_usd: float | None = None,
    strategy_name: str = "Agresif",
) -> SimulationResult:
    return _simulate_predictions(
        _model_1_predictions(market),
        gold_ohlc,
        "Model 1 - Harga Historis",
        initial_balance,
        lot_size,
        take_profit_usd,
        swap_per_position,
        max_buy_positions,
        max_sell_positions,
        entry_threshold_pct,
        stop_loss_usd,
        strategy_name,
    )


def simulate_model_2(
    market: pd.DataFrame,
    gold_ohlc: pd.DataFrame,
    initial_balance: float = 1000.0,
    lot_size: float = 0.01,
    take_profit_usd: float = 5.0,
    swap_per_position: float = 0.2,
    max_buy_positions: int = 8,
    max_sell_positions: int = 10,
    entry_threshold_pct: float = 0.0,
    stop_loss_usd: float | None = None,
    strategy_name: str = "Agresif",
) -> SimulationResult:
    return _simulate_predictions(
        _model_2_predictions(market),
        gold_ohlc,
        "Model 2 - Lintas Pasar",
        initial_balance,
        lot_size,
        take_profit_usd,
        swap_per_position,
        max_buy_positions,
        max_sell_positions,
        entry_threshold_pct,
        stop_loss_usd,
        strategy_name,
    )


def run_simulation_scenarios(
    market: pd.DataFrame,
    gold_ohlc: pd.DataFrame,
    scenarios: list[dict[str, float | str]] | None = None,
) -> tuple[dict[str, dict[str, SimulationResult]], pd.DataFrame]:
    scenario_rows: list[dict[str, object]] = []
    results: dict[str, dict[str, SimulationResult]] = {
        "Model 1 - Harga Historis": {},
        "Model 2 - Lintas Pasar": {},
    }
    model_predictions = {
        "Model 1 - Harga Historis": _model_1_predictions(market),
        "Model 2 - Lintas Pasar": _model_2_predictions(market),
    }

    for scenario in scenarios or DEFAULT_SCENARIOS:
        strategy_name = str(scenario["Strategi"])
        for model_name, predictions in model_predictions.items():
            result = _simulate_predictions(
                predictions,
                gold_ohlc,
                model_name,
                take_profit_usd=float(scenario["take_profit_usd"]),
                entry_threshold_pct=float(scenario["entry_threshold_pct"]),
                stop_loss_usd=float(scenario["stop_loss_usd"]),
                strategy_name=strategy_name,
            )
            results[model_name][strategy_name] = result
            summary = result.summary
            scenario_rows.append(
                {
                    "Model": model_name,
                    "Strategi": strategy_name,
                    "Threshold entry (%)": float(scenario["entry_threshold_pct"]),
                    "TP (USD)": float(scenario["take_profit_usd"]),
                    "SL (USD)": float(scenario["stop_loss_usd"]),
                    "Balance akhir": summary["Balance akhir"],
                    "Total net P/L": summary["Total net P/L"],
                    "Jumlah transaksi": summary["Jumlah transaksi"],
                    "Win rate": summary["Win rate"],
                    "Max drawdown": summary["Max drawdown"],
                    "Profit factor": summary["Profit factor"],
                    "Avg net P/L": summary["Avg net P/L"],
                }
            )

    return results, pd.DataFrame(scenario_rows)
