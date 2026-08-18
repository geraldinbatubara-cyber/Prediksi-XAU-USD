from __future__ import annotations

import base64
import os
import pickle
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from gold_forecast.paper_ledger_store import configure_paper_ledger_store
from gold_forecast.supabase_broker import load_supabase_broker_feed


SIMULATION_PATH = Path("data/precomputed/simulations.pkl")
SIMULATION_VERSION = "optimizer-v1-only-2025q1-2026q2"
BUY_SPECIALIST_MODEL_PATH = Path(
    "data/precomputed/buy_specialist_v4_live.pkl.b64"
)
BUY_SPECIALIST_MODEL_VERSION = "buy-specialist-v4-live-inference-2026-07-24-v1"
SIDEWAYS_MODERATE_MODEL_PATH = Path(
    "data/precomputed/sideways_moderate_live.pkl.b64"
)
SIDEWAYS_MODERATE_MODEL_VERSION = (
    "sideways-v9-moderate-regime-live-inference-2026-08-17-v1"
)


def _load_v1_leaderboard() -> pd.DataFrame:
    with SIMULATION_PATH.open("rb") as file:
        saved = pickle.load(file)
    if saved.get("version") != SIMULATION_VERSION:
        raise RuntimeError("Versi artefak Optimizer v1 tidak sesuai dengan kontrak live.")
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
    base_url = os.getenv("SUPABASE_URL", "").strip()
    write_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    read_key = (
        os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
        or write_key
    )
    symbol = os.getenv("SUPABASE_SYMBOL", "XAUUSD").strip() or "XAUUSD"
    return base_url, read_key, write_key, symbol


def _print_result(label: str, path: Path, result: dict[str, object]) -> None:
    summary = result["summary"]
    print(f"[{label}] ledger={path} rows={len(result['ledger'])}")
    print(
        f"[{label}] equity={summary['Equity']:.2f} | "
        f"open BUY={summary['Open BUY']} | open SELL={summary['Open SELL']}"
    )
    print(
        f"[{label}] last update WIT="
        f"{summary['Now WIT'].strftime('%Y-%m-%d %H:%M:%S WIT')}"
    )


def _run_and_report(label: str, path: Path, **kwargs) -> str | None:
    try:
        result = run_live_trading_update(path=path, **kwargs)
        _print_result(label, path, result)
        return None
    except Exception as exc:
        print(f"ERROR [{label}]: {exc}")
        return f"{label}: {exc}"


def main() -> None:
    gold_ohlc = load_gold_data()
    leaderboard = _load_v1_leaderboard()
    base_url, read_key, write_key, symbol = _supabase_config()
    broker_bars = pd.DataFrame()
    broker_quote = None
    if base_url and read_key and write_key:
        configure_paper_ledger_store(base_url, read_key, write_key)
        try:
            broker_bars, broker_quotes = load_supabase_broker_feed(
                base_url,
                read_key,
                symbol=symbol,
                bars_limit=5000,
            )
            if not broker_quotes.empty:
                broker_quote = broker_quotes.iloc[-1]
        except Exception as exc:
            print(f"WARNING: Feed broker Supabase gagal dimuat: {exc}")

    failures = []
    error = _run_and_report(
        "Baseline v1",
        path=LIVE_TRADING_PATH,
        gold_ohlc=gold_ohlc,
        optimizer_leaderboard=leaderboard,
        start_date=LIVE_START_DATE,
        broker_quote=broker_quote,
    )
    if error:
        failures.append(error)

    if not (base_url and read_key and write_key):
        print(
            "WARNING: Supabase GitHub secrets belum lengkap. Baseline v1 tetap "
            "diperbarui, tetapi tiga strategi berbasis feed broker dilewati."
        )
        return

    if broker_quote is None:
        print("WARNING: Quote broker Supabase kosong; strategi M1 dilewati.")
        if failures:
            raise RuntimeError(" | ".join(failures))
        return

    error = _run_and_report(
        "Fixed Delay 5m",
        path=LIVE_TRADING_FIXED_DELAY_PATH,
        gold_ohlc=gold_ohlc,
        optimizer_leaderboard=leaderboard,
        start_date=LIVE_FIXED_DELAY_START,
        broker_quote=broker_quote,
        entry_strategy="fixed_delay_5m",
        broker_bars=broker_bars,
    )
    if error:
        failures.append(error)

    try:
        buy_bundle = _load_model_bundle(
            BUY_SPECIALIST_MODEL_PATH,
            BUY_SPECIALIST_MODEL_VERSION,
        )
        error = _run_and_report(
            "BUY Specialist v4",
            path=LIVE_TRADING_BUY_SPECIALIST_V4_PATH,
            gold_ohlc=gold_ohlc,
            optimizer_leaderboard=leaderboard,
            start_date=LIVE_BUY_SPECIALIST_V4_START,
            broker_quote=broker_quote,
            entry_strategy="buy_specialist_v4",
            broker_bars=broker_bars,
            strategy_model_bundle=buy_bundle,
        )
    except Exception as exc:
        error = f"BUY Specialist v4: {exc}"
        print(f"ERROR [{error}]")
    if error:
        failures.append(error)

    try:
        sideways_bundle = _load_model_bundle(
            SIDEWAYS_MODERATE_MODEL_PATH,
            SIDEWAYS_MODERATE_MODEL_VERSION,
        )
        error = _run_and_report(
            "Moderate Regime",
            path=LIVE_TRADING_SIDEWAYS_MODERATE_PATH,
            gold_ohlc=gold_ohlc,
            optimizer_leaderboard=leaderboard,
            start_date=LIVE_SIDEWAYS_MODERATE_START,
            broker_quote=broker_quote,
            entry_strategy="sideways_moderate",
            broker_bars=broker_bars,
            strategy_model_bundle=sideways_bundle,
        )
    except Exception as exc:
        error = f"Moderate Regime: {exc}"
        print(f"ERROR [{error}]")
    if error:
        failures.append(error)

    if failures:
        raise RuntimeError("Sebagian strategi gagal diperbarui: " + " | ".join(failures))


if __name__ == "__main__":
    main()
