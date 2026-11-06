from __future__ import annotations

import argparse
import base64
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.broker_data import (
    BROKER_BARS_PATH,
    BROKER_QUOTE_PATH,
    load_broker_bars,
    load_broker_quote,
)
from gold_forecast.data import load_gold_data, refresh_gold_cache
from gold_forecast.live_trading import (
    LIVE_BUY_SPECIALIST_V4_START,
    LIVE_FIXED_DELAY_START,
    LIVE_SIDEWAYS_MODERATE_START,
    LIVE_START_DATE,
    LIVE_TRADING_BUY_SPECIALIST_V4_PATH,
    LIVE_TRADING_FIXED_DELAY_PATH,
    LIVE_TRADING_PATH,
    LIVE_TRADING_SIDEWAYS_MODERATE_PATH,
    run_live_trading_update,
)
from gold_forecast.paper_ledger_store import (
    configure_paper_ledger_store,
    save_strategy_state,
    strategy_id_for_path,
)
from gold_forecast.supabase_broker import load_supabase_broker_feed


SIMULATION_PATH = PROJECT_ROOT / "data" / "precomputed" / "simulations.pkl"
SIMULATION_VERSION = "optimizer-v1-only-2025q1-2026q2"
BUY_SPECIALIST_MODEL_PATH = (
    PROJECT_ROOT / "data" / "precomputed" / "buy_specialist_v4_live.pkl.b64"
)
BUY_SPECIALIST_MODEL_VERSION = "buy-specialist-v4-live-inference-2026-07-24-v1"
SIDEWAYS_MODERATE_MODEL_PATH = (
    PROJECT_ROOT / "data" / "precomputed" / "sideways_moderate_live.pkl.b64"
)
SIDEWAYS_MODERATE_MODEL_VERSION = (
    "sideways-v9-moderate-regime-live-inference-2026-08-17-v1"
)
STATE_KEYS = (
    "summary",
    "params",
    "signal",
    "historical_signal",
    "decision_snapshot",
    "decision_persisted",
    "waiting_state",
    "trigger_state",
    "fixed_delay_state",
    "specialist_state",
)
GOLD_REFRESH_INTERVAL_SECONDS = 3 * 60 * 60
GOLD_REFRESH_RETRY_SECONDS = 15 * 60


def _load_v1_leaderboard() -> pd.DataFrame:
    with SIMULATION_PATH.open("rb") as file:
        saved = pickle.load(file)
    if saved.get("version") != SIMULATION_VERSION:
        raise RuntimeError("Versi artefak Optimizer v1 tidak sesuai kontrak live.")
    payload = saved.get("payload")
    if not isinstance(payload, tuple) or len(payload) < 2:
        raise RuntimeError("Artefak Optimizer v1 tidak memiliki leaderboard.")
    leaderboard = payload[1]
    if not isinstance(leaderboard, pd.DataFrame) or leaderboard.empty:
        raise RuntimeError("Leaderboard Optimizer v1 kosong.")
    return leaderboard


def _load_model_bundle(path: Path, expected_version: str) -> dict[str, object]:
    saved = pickle.loads(base64.b64decode(path.read_text(encoding="ascii")))
    if saved.get("version") != expected_version:
        raise RuntimeError(f"Versi artefak {path.name} tidak sesuai kontrak live.")
    payload = saved.get("payload")
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"Artefak {path.name} tidak memiliki model live.")
    return payload


def _supabase_config() -> tuple[str, str, str, str]:
    base_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    write_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    read_key = (
        os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
        or write_key
    )
    symbol = os.getenv("SUPABASE_SYMBOL", "XAUUSD").strip() or "XAUUSD"
    if not base_url or not read_key or not write_key:
        raise RuntimeError("Konfigurasi Supabase worker belum lengkap.")
    configure_paper_ledger_store(base_url, read_key, write_key)
    return base_url, read_key, write_key, symbol


def _strategy_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "Baseline v1",
            "path": LIVE_TRADING_PATH,
            "start_date": LIVE_START_DATE,
            "entry_strategy": "baseline",
            "model": None,
        },
        {
            "name": "Fixed Delay 5m",
            "path": LIVE_TRADING_FIXED_DELAY_PATH,
            "start_date": LIVE_FIXED_DELAY_START,
            "entry_strategy": "fixed_delay_5m",
            "model": None,
        },
        {
            "name": "BUY Specialist v4",
            "path": LIVE_TRADING_BUY_SPECIALIST_V4_PATH,
            "start_date": LIVE_BUY_SPECIALIST_V4_START,
            "entry_strategy": "buy_specialist_v4",
            "model": _load_model_bundle(
                BUY_SPECIALIST_MODEL_PATH,
                BUY_SPECIALIST_MODEL_VERSION,
            ),
        },
        {
            "name": "Moderate Regime",
            "path": LIVE_TRADING_SIDEWAYS_MODERATE_PATH,
            "start_date": LIVE_SIDEWAYS_MODERATE_START,
            "entry_strategy": "sideways_moderate",
            "model": _load_model_bundle(
                SIDEWAYS_MODERATE_MODEL_PATH,
                SIDEWAYS_MODERATE_MODEL_VERSION,
            ),
        },
    ]


def _broker_snapshot(
    base_url: str,
    read_key: str,
    symbol: str,
) -> tuple[pd.DataFrame, pd.Series | None]:
    local_bars = load_broker_bars(BROKER_BARS_PATH)
    local_quotes = load_broker_quote(BROKER_QUOTE_PATH)
    if not local_bars.empty and not local_quotes.empty:
        return local_bars, local_quotes.iloc[-1]

    try:
        bars, quotes = load_supabase_broker_feed(
            base_url,
            read_key,
            symbol=symbol,
            bars_limit=5000,
        )
        if not bars.empty or not quotes.empty:
            return bars, (quotes.iloc[-1] if not quotes.empty else None)
    except Exception as exc:
        print(f"WARNING: Feed broker Supabase gagal dimuat: {exc}", flush=True)

    return local_bars, (local_quotes.iloc[-1] if not local_quotes.empty else None)


def _refresh_gold_snapshot() -> tuple[pd.DataFrame, bool]:
    try:
        gold_ohlc = refresh_gold_cache()
        latest_date = pd.Timestamp(gold_ohlc.index.max()).strftime("%Y-%m-%d")
        print(f"GC=F daily cache refreshed through {latest_date}.", flush=True)
        return gold_ohlc, True
    except Exception as exc:
        print(
            f"WARNING: GC=F daily cache refresh failed; using existing cache: {exc}",
            flush=True,
        )
        return load_gold_data(), False


def run_cycle(
    leaderboard: pd.DataFrame,
    strategies: list[dict[str, object]],
    base_url: str,
    read_key: str,
    symbol: str,
    gold_ohlc: pd.DataFrame,
) -> bool:
    broker_bars, broker_quote = _broker_snapshot(base_url, read_key, symbol)
    cycle_ok = True

    for strategy in strategies:
        try:
            result = run_live_trading_update(
                gold_ohlc,
                leaderboard,
                path=strategy["path"],
                start_date=strategy["start_date"],
                broker_quote=broker_quote,
                entry_strategy=strategy["entry_strategy"],
                broker_bars=broker_bars,
                strategy_model_bundle=strategy["model"],
            )
            state = {key: result.get(key) for key in STATE_KEYS}
            state["worker_name"] = "Windows Paper Trading Worker"
            if not save_strategy_state(strategy_id_for_path(strategy["path"]), state):
                raise RuntimeError("snapshot strategi gagal disimpan ke Supabase")
            summary = result["summary"]
            print(
                f"{strategy['name']}: equity={summary['Equity']:.2f}, "
                f"open={summary['Open BUY'] + summary['Open SELL']}, "
                f"updated={summary['Now WIT'].strftime('%Y-%m-%d %H:%M:%S WIT')}",
                flush=True,
            )
        except Exception as exc:
            cycle_ok = False
            print(f"ERROR {strategy['name']}: {exc}", file=sys.stderr, flush=True)
    return cycle_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Perbarui empat strategi paper trading.")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    base_url, read_key, _, symbol = _supabase_config()
    leaderboard = _load_v1_leaderboard()
    strategies = _strategy_specs()
    gold_ohlc, refresh_ok = _refresh_gold_snapshot()
    next_gold_refresh = time.monotonic() + (
        GOLD_REFRESH_INTERVAL_SECONDS if refresh_ok else GOLD_REFRESH_RETRY_SECONDS
    )
    if not run_cycle(leaderboard, strategies, base_url, read_key, symbol, gold_ohlc):
        raise RuntimeError("Siklus awal worker gagal untuk satu atau lebih strategi.")
    print("PAPER_WORKER_READY", flush=True)

    if args.once:
        return
    while True:
        time.sleep(max(args.interval, 10))
        if time.monotonic() >= next_gold_refresh:
            gold_ohlc, refresh_ok = _refresh_gold_snapshot()
            next_gold_refresh = time.monotonic() + (
                GOLD_REFRESH_INTERVAL_SECONDS
                if refresh_ok
                else GOLD_REFRESH_RETRY_SECONDS
            )
        run_cycle(leaderboard, strategies, base_url, read_key, symbol, gold_ohlc)


if __name__ == "__main__":
    main()
