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


def _enum_name(value: object, mapping: dict[int, str], default: str = "UNKNOWN") -> str:
    try:
        return mapping.get(int(value), default)
    except (TypeError, ValueError):
        return default


def _terminal_status(mt5, symbol: str, received_at_utc: pd.Timestamp) -> dict[str, object]:
    account = mt5.account_info()
    terminal = mt5.terminal_info()
    specification = mt5.symbol_info(symbol)
    if account is None or terminal is None or specification is None:
        raise RuntimeError(f"Status akun/terminal/simbol tidak tersedia: {mt5.last_error()}")

    account_mode = _enum_name(
        account.trade_mode,
        {
            int(mt5.ACCOUNT_TRADE_MODE_DEMO): "DEMO",
            int(mt5.ACCOUNT_TRADE_MODE_CONTEST): "CONTEST",
            int(mt5.ACCOUNT_TRADE_MODE_REAL): "REAL",
        },
    )
    symbol_trade_mode = _enum_name(
        specification.trade_mode,
        {
            int(mt5.SYMBOL_TRADE_MODE_DISABLED): "DISABLED",
            int(mt5.SYMBOL_TRADE_MODE_LONGONLY): "LONG_ONLY",
            int(mt5.SYMBOL_TRADE_MODE_SHORTONLY): "SHORT_ONLY",
            int(mt5.SYMBOL_TRADE_MODE_CLOSEONLY): "CLOSE_ONLY",
            int(mt5.SYMBOL_TRADE_MODE_FULL): "FULL",
        },
    )
    return {
        "symbol": symbol,
        "account_mode": account_mode,
        "broker_server": str(account.server),
        "broker_company": str(account.company),
        "terminal_connected": bool(terminal.connected),
        "account_trade_allowed": bool(account.trade_allowed),
        "terminal_trade_allowed": bool(terminal.trade_allowed),
        "symbol_trade_mode": symbol_trade_mode,
        "leverage": int(account.leverage),
        "contract_size": float(specification.trade_contract_size),
        "volume_min": float(specification.volume_min),
        "volume_max": float(specification.volume_max),
        "volume_step": float(specification.volume_step),
        "stops_level_points": int(specification.trade_stops_level),
        "spread_points": int(specification.spread),
        "spread_is_floating": bool(specification.spread_float),
        "swap_long": float(specification.swap_long),
        "swap_short": float(specification.swap_short),
        "currency": str(account.currency),
        "manual_execution_only": True,
        "updated_at": received_at_utc,
    }


def _write_snapshot(
    mt5,
    symbol: str,
    bars_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
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
    received_at_utc = pd.Timestamp.now(tz="UTC")
    terminal_status = _terminal_status(mt5, symbol, received_at_utc)
    source = f"MT5 {terminal_status['account_mode']}"
    bars["source"] = source
    bars[
        ["timestamp_utc", "open", "high", "low", "close", "tick_volume", "spread_points", "symbol", "source"]
    ].to_csv(BROKER_BARS_PATH, index=False)

    quote = pd.DataFrame(
        [
            {
                "timestamp_utc": pd.to_datetime(tick.time_msc, unit="ms", utc=True),
                "received_at_utc": received_at_utc,
                "bid": float(tick.bid),
                "ask": float(tick.ask),
                "symbol": symbol,
                "source": source,
            }
        ]
    )
    quote.to_csv(BROKER_QUOTE_PATH, index=False)
    print(
        f"{quote.iloc[0]['timestamp_utc']} | {symbol} | bid={tick.bid:.5f} | "
        f"ask={tick.ask:.5f} | bars={len(bars)}"
    )
    return bars, quote, terminal_status


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
            bars, quote, terminal_status = _write_snapshot(mt5, args.symbol, args.bars)
            if args.publish_supabase:
                publish_bars = bars if first_publish else bars.tail(5)
                try:
                    publish_broker_snapshot(
                        supabase_url,
                        service_role_key,
                        publish_bars,
                        quote,
                        terminal_status=terminal_status,
                    )
                    print(
                        f"Supabase updated | quote=1 | bars={len(publish_bars)} | "
                        f"account={terminal_status['account_mode']}"
                    )
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
