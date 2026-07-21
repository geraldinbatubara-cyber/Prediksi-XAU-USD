from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.broker_data import BROKER_BARS_PATH, BROKER_QUOTE_PATH
from gold_forecast.supabase_broker import publish_broker_snapshot


def _load_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("Paket MetaTrader5 belum terpasang. Jalankan: pip install MetaTrader5") from exc
    return mt5


def _write_snapshot(mt5, symbol: str, bars_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Simbol {symbol} tidak tersedia di Market Watch MT5.")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Tick {symbol} belum tersedia: {mt5.last_error()}")

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, bars_count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Candle M1 {symbol} belum tersedia: {mt5.last_error()}")

    BROKER_BARS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bars = pd.DataFrame(rates).rename(columns={"time": "timestamp_utc", "spread": "spread_points"})
    bars["timestamp_utc"] = pd.to_datetime(bars["timestamp_utc"], unit="s", utc=True)
    bars["symbol"] = symbol
    bars["source"] = "MT5 Demo"
    bars[
        ["timestamp_utc", "open", "high", "low", "close", "tick_volume", "spread_points", "symbol", "source"]
    ].to_csv(BROKER_BARS_PATH, index=False)

    quote = pd.DataFrame(
        [
            {
                "timestamp_utc": pd.to_datetime(tick.time_msc, unit="ms", utc=True),
                "bid": float(tick.bid),
                "ask": float(tick.ask),
                "symbol": symbol,
                "source": "MT5 Demo",
            }
        ]
    )
    quote.to_csv(BROKER_QUOTE_PATH, index=False)
    print(
        f"{quote.iloc[0]['timestamp_utc']} | {symbol} | bid={tick.bid:.5f} | "
        f"ask={tick.ask:.5f} | bars={len(bars)}"
    )
    return bars, quote


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge read-only MT5 untuk data XAUUSD.")
    parser.add_argument("--symbol", default="XAUUSD", help="Nama simbol persis seperti di Market Watch broker.")
    parser.add_argument("--bars", type=int, default=3000, help="Jumlah candle M1 yang disalin.")
    parser.add_argument("--interval", type=int, default=60, help="Interval pembaruan dalam detik.")
    parser.add_argument("--once", action="store_true", help="Ambil satu snapshot lalu berhenti.")
    parser.add_argument(
        "--publish-supabase",
        action="store_true",
        help="Kirim feed ke Supabase memakai SUPABASE_URL dan SUPABASE_SERVICE_ROLE_KEY.",
    )
    args = parser.parse_args()

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if args.publish_supabase and (not supabase_url or not service_role_key):
        raise RuntimeError(
            "Set SUPABASE_URL dan SUPABASE_SERVICE_ROLE_KEY di environment sebelum memakai --publish-supabase."
        )

    mt5 = _load_mt5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 gagal diinisialisasi: {mt5.last_error()}")
    first_publish = True
    try:
        while True:
            bars, quote = _write_snapshot(mt5, args.symbol, args.bars)
            if args.publish_supabase:
                publish_bars = bars if first_publish else bars.tail(5)
                try:
                    publish_broker_snapshot(supabase_url, service_role_key, publish_bars, quote)
                    print(f"Supabase updated | quote=1 | bars={len(publish_bars)}")
                except Exception as exc:
                    print(f"WARNING Supabase: {exc}", file=sys.stderr)
                    if args.once:
                        raise
                first_publish = False
            if args.once:
                break
            time.sleep(max(args.interval, 5))
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
