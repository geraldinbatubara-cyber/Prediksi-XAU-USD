from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from datetime import timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.broker_data import apply_broker_clock_offset, load_broker_bars, load_broker_quote
from gold_forecast.m1_backtest import run_fixed_v1_intraday_history, run_intraday_optimization


OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "m1_backtests.pkl"
HISTORY_DIR = PROJECT_ROOT / "data" / "intraday"
VERSION = "v1-full-history-v10-oos-2026-07-21"
REQUESTED_START = pd.Timestamp("2025-01-01")
REQUESTED_END = pd.Timestamp("2026-06-30 23:59:59")


def _load_mt5_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 gagal diinisialisasi: {mt5.last_error()}")
    try:
        if not mt5.symbol_select("XAUUSD", True):
            raise RuntimeError(f"XAUUSD tidak tersedia: {mt5.last_error()}")
        tick = mt5.symbol_info_tick("XAUUSD")
        if tick is None:
            raise RuntimeError(f"Tick XAUUSD tidak tersedia: {mt5.last_error()}")
        quote = _quote_frame(tick)
        monthly_frames = []
        for period in pd.period_range(REQUESTED_START.to_period("M"), REQUESTED_END.to_period("M"), freq="M"):
            path = HISTORY_DIR / f"xauusd_m1_{period}.csv.gz"
            if path.exists():
                month = pd.read_csv(path, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
                print(f"Cached {period}: {len(month)} bars")
            else:
                start = (period.start_time - pd.Timedelta(days=1)).tz_localize(timezone.utc).to_pydatetime()
                end = ((period + 1).start_time + pd.Timedelta(days=1)).tz_localize(timezone.utc).to_pydatetime()
                rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M1, start, end)
                if rates is None or not len(rates):
                    raise RuntimeError(f"Histori M1 {period} tidak tersedia: {mt5.last_error()}")
                month = _normalize_rates(pd.DataFrame(rates), quote, period)
                if month.empty:
                    raise RuntimeError(f"Histori M1 {period} kosong setelah normalisasi waktu broker.")
                HISTORY_DIR.mkdir(parents=True, exist_ok=True)
                month.reset_index().to_csv(path, index=False, compression="gzip")
                print(f"Downloaded {period}: {len(month)} bars")
            monthly_frames.append(month)

        daily_start = (REQUESTED_START - pd.Timedelta(days=180)).tz_localize(timezone.utc).to_pydatetime()
        daily_end = (REQUESTED_END + pd.Timedelta(days=2)).tz_localize(timezone.utc).to_pydatetime()
        daily_rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_D1, daily_start, daily_end)
        if daily_rates is None or not len(daily_rates):
            raise RuntimeError(f"Histori D1 XAUUSD tidak tersedia: {mt5.last_error()}")
    finally:
        mt5.shutdown()

    gold_m1 = pd.concat(monthly_frames).sort_index()
    gold_m1 = gold_m1.loc[~gold_m1.index.duplicated(keep="last")]
    daily = pd.DataFrame(daily_rates).rename(columns={"time": "timestamp_utc"})
    daily["timestamp_utc"] = pd.to_datetime(daily["timestamp_utc"], unit="s", utc=True)
    offset = float(quote.iloc[-1]["clock_offset_hours"])
    daily["timestamp_utc"] = daily["timestamp_utc"] - pd.to_timedelta(offset, unit="h")
    daily = daily.set_index("timestamp_utc")[["open", "high", "low", "close", "tick_volume"]]
    daily.index = daily.index.tz_convert("UTC").tz_localize(None).normalize()
    daily = daily.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"})
    return gold_m1, daily


def _quote_frame(tick) -> pd.DataFrame:
    received_at = pd.Timestamp.now(tz="UTC")
    raw = pd.DataFrame([{
        "timestamp_utc": pd.to_datetime(tick.time_msc, unit="ms", utc=True),
        "received_at_utc": received_at,
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "symbol": "XAUUSD",
        "source": "MT5 DEMO",
    }])
    return load_broker_quote(raw)


def _normalize_rates(rates: pd.DataFrame, quote: pd.DataFrame, period: pd.Period) -> pd.DataFrame:
    bars = pd.DataFrame(rates).rename(columns={"time": "timestamp_utc", "spread": "spread_points"})
    bars["timestamp_utc"] = pd.to_datetime(bars["timestamp_utc"], unit="s", utc=True)
    bars["symbol"] = "XAUUSD"
    bars["source"] = "MT5 DEMO"
    clean_bars = load_broker_bars(bars)
    clean_bars, _ = apply_broker_clock_offset(clean_bars, quote)
    indexed = clean_bars.set_index("timestamp_utc")[["open", "high", "low", "close", "spread_points"]]
    indexed.index = indexed.index.tz_convert("UTC").tz_localize(None)
    indexed = indexed.loc[(indexed.index >= period.start_time) & (indexed.index <= period.end_time)]
    return indexed.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "spread_points": "SpreadPoints",
    })


def _compact(result):
    curve = result.equity_curve
    compact_curve = curve
    if len(curve) > 5000:
        important = pd.concat([
            curve.loc[[curve["Equity"].idxmin()]],
            curve.loc[[curve["Equity"].idxmax()]],
            curve.tail(1),
        ])
        compact_curve = pd.concat([curve.iloc[::30], important]).sort_index()
        compact_curve = compact_curve.loc[~compact_curve.index.duplicated(keep="last")]
    return replace(result, trades=result.trades.tail(1000).copy(), equity_curve=compact_curve)


def main() -> None:
    previous_payload = None
    if OUTPUT_PATH.exists():
        try:
            previous_payload = pickle.load(OUTPUT_PATH.open("rb")).get("payload")
        except Exception:
            previous_payload = None
    gold_m1, gold_daily = _load_mt5_history()
    v1 = run_fixed_v1_intraday_history(
        gold_m1, gold_daily, requested_start=REQUESTED_START, requested_end=REQUESTED_END
    )
    if previous_payload is not None and len(previous_payload) >= 4:
        v10 = (previous_payload[2], previous_payload[3])
    else:
        recent = gold_m1.loc[gold_m1.index >= pd.Timestamp("2026-04-01")]
        v10 = run_intraday_optimization(
            recent, gold_daily, variant="v10", requested_start=REQUESTED_START, requested_end=REQUESTED_END
        )
    payload = (_compact(v1[0]), v1[1], _compact(v10[0]), v10[1])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as file:
        pickle.dump({"version": VERSION, "payload": payload}, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {OUTPUT_PATH}")
    summary = v1[0].summary
    print(
        f"v1 Intraday M1 extended: {summary['Periode uji']} | equity={summary['Equity akhir']:.2f} | "
        f"growth={summary['Growth total']:+.2f}% | trades={summary['Jumlah transaksi']:.0f}"
    )
    print(summary["Pertumbuhan bulanan"][["Bulan", "Equity akhir", "Growth bulanan (%)", "Growth kumulatif (%)"]].to_string(index=False))


if __name__ == "__main__":
    main()
