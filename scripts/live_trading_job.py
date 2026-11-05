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
from gold_forecast.data import load_gold_data
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


SIMULATION_PATH = PROJECT_ROOT / "data" / "precomputed" / "simulations.pkl"
BUY_SPECIALIST_MODEL_PATH = (
    PROJECT_ROOT / "data" / "precomputed" / "buy_specialist_v4_live.pkl.b64"
)
SIDEWAYS_MODERATE_MODEL_PATH = (
    PROJECT_ROOT / "data" / "precomputed" / "sideways_moderate_live.pkl.b64"
)
STATE_KEYS = (
    "summary",
    "params",
    "signal",
    "waiting_state",
    "trigger_state",
    "fixed_delay_state",
    "specialist_state",
)


def _load_leaderboard() -> pd.DataFrame:
    with SIMULATION_PATH.open("rb") as file:
        saved = pickle.load(file)
    payload = saved.get("payload", ())
    if len(payload) < 2 or not isinstance(payload[1], pd.DataFrame):
        raise RuntimeError("Leaderboard Optimizer v1 precomputed tidak tersedia.")
    return payload[1]


def _load_model(path: Path) -> dict[str, object]:
    saved = pickle.loads(base64.b64decode(path.read_bytes()))
    payload = saved.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Artefak model tidak valid: {path}")
    return payload


def _configure_store() -> None:
    base_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not base_url or not service_key:
        raise RuntimeError("SUPABASE_URL dan SUPABASE_SERVICE_ROLE_KEY wajib tersedia.")
    configure_paper_ledger_store(base_url, service_key, service_key)


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
            "model": _load_model(BUY_SPECIALIST_MODEL_PATH),
        },
        {
            "name": "Moderate Regime",
            "path": LIVE_TRADING_SIDEWAYS_MODERATE_PATH,
            "start_date": LIVE_SIDEWAYS_MODERATE_START,
            "entry_strategy": "sideways_moderate",
            "model": _load_model(SIDEWAYS_MODERATE_MODEL_PATH),
        },
    ]


def run_cycle(
    leaderboard: pd.DataFrame,
    strategies: list[dict[str, object]],
) -> bool:
    gold_ohlc = load_gold_data()
    broker_bars = load_broker_bars(BROKER_BARS_PATH)
    quotes = load_broker_quote(BROKER_QUOTE_PATH)
    broker_quote = quotes.iloc[-1] if not quotes.empty else None
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

    _configure_store()
    leaderboard = _load_leaderboard()
    strategies = _strategy_specs()
    if not run_cycle(leaderboard, strategies):
        raise RuntimeError("Siklus awal worker gagal untuk satu atau lebih strategi.")
    print("PAPER_WORKER_READY", flush=True)

    if args.once:
        return
    while True:
        time.sleep(max(args.interval, 10))
        run_cycle(leaderboard, strategies)


if __name__ == "__main__":
    main()
