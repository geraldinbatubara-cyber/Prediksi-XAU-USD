from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.broker_data import apply_broker_clock_offset, load_broker_bars, load_broker_quote
from gold_forecast.m1_backtest import run_fixed_m1_strategy


OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "m1_backtests.pkl"
VERSION = "fixed-v1-v10-m1-2026-07-21"
REQUESTED_START = pd.Timestamp("2025-01-01")
REQUESTED_END = pd.Timestamp("2026-06-30 23:59:59")

V1_PARAMS = {
    "Mode": "Trend", "Strategi": "Trend | MA 10/50 | Mom 10 | TP 25 SL 10 | Lot 0.01",
    "Fast MA": 10, "Slow MA": 50, "Momentum hari": 10, "Threshold entry (%)": 0.15,
    "TP (USD)": 25.0, "SL (USD)": 10.0, "Lot": 0.01, "Target fase (%)": 20.0,
    "Close-all target equity": True,
}
V10_PARAMS = {
    "Mode": "Trend",
    "Strategi": "Trend | MA 20/50 | Mom 14 | TP 50 SL 18 | Lot 0.03 | Max 8/10 | Protection 60/40/20",
    "Fast MA": 20, "Slow MA": 50, "Momentum hari": 14, "Threshold entry (%)": 0.10,
    "TP (USD)": 50.0, "SL (USD)": 18.0, "Lot": 0.03, "Max BUY": 8, "Max SELL": 10,
    "Risk cap floating SL (%)": 50.0, "Target fase (%)": 20.0,
    "Profit protection aktif (USD)": 60.0, "Profit protection floor (USD)": 40.0,
    "Profit protection trail (USD)": 20.0, "Close-all target equity": False,
}


def _load_mt5_m1() -> pd.DataFrame:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 gagal diinisialisasi: {mt5.last_error()}")
    try:
        if not mt5.symbol_select("XAUUSD", True):
            raise RuntimeError(f"XAUUSD tidak tersedia: {mt5.last_error()}")
        terminal = mt5.terminal_info()
        count = max(3000, int(getattr(terminal, "maxbars", 100000)) - 1)
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, count)
        tick = mt5.symbol_info_tick("XAUUSD")
        if rates is None or not len(rates) or tick is None:
            raise RuntimeError(f"Histori M1 tidak tersedia: {mt5.last_error()}")
    finally:
        mt5.shutdown()

    received_at = pd.Timestamp.now(tz="UTC")
    bars = pd.DataFrame(rates).rename(columns={"time": "timestamp_utc", "spread": "spread_points"})
    bars["timestamp_utc"] = pd.to_datetime(bars["timestamp_utc"], unit="s", utc=True)
    bars["symbol"] = "XAUUSD"
    bars["source"] = "MT5 DEMO"
    quote = pd.DataFrame([{
        "timestamp_utc": pd.to_datetime(tick.time_msc, unit="ms", utc=True),
        "received_at_utc": received_at, "bid": float(tick.bid), "ask": float(tick.ask),
        "symbol": "XAUUSD", "source": "MT5 DEMO",
    }])
    clean_bars = load_broker_bars(bars)
    clean_quote = load_broker_quote(quote)
    clean_bars, _ = apply_broker_clock_offset(clean_bars, clean_quote)
    indexed = clean_bars.set_index("timestamp_utc")[["open", "high", "low", "close"]]
    indexed.index = indexed.index.tz_convert("UTC").tz_localize(None)
    return indexed.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})


def _compact(result):
    curve = result.equity_curve
    compact_curve = curve
    if len(curve) > 5000:
        important = pd.concat([
            curve.loc[[curve["Equity"].idxmin()]], curve.loc[[curve["Equity"].idxmax()]], curve.tail(1)
        ])
        compact_curve = pd.concat([curve.iloc[::60], important]).sort_index()
        compact_curve = compact_curve.loc[~compact_curve.index.duplicated(keep="last")]
    return replace(result, trades=result.trades.tail(1000).copy(), equity_curve=compact_curve)


def main() -> None:
    gold_m1 = _load_mt5_m1()
    v1 = run_fixed_m1_strategy(gold_m1, V1_PARAMS, model_name="Optimizer v1 M1", requested_start=REQUESTED_START, requested_end=REQUESTED_END)
    v10 = run_fixed_m1_strategy(gold_m1, V10_PARAMS, model_name="Optimizer v10 M1", requested_start=REQUESTED_START, requested_end=REQUESTED_END)
    payload = (_compact(v1[0]), v1[1], _compact(v10[0]), v10[1])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as file:
        pickle.dump({"version": VERSION, "payload": payload}, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {OUTPUT_PATH}")
    for name, result in [("v1 M1", v1[0]), ("v10 M1", v10[0])]:
        print(f"{name}: {result.summary['Periode uji']} | candles={result.summary['Jumlah candle']:.0f} | equity={result.summary['Equity akhir']:.2f} | trades={result.summary['Jumlah transaksi']:.0f}")


if __name__ == "__main__":
    main()
