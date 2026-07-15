from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.data import load_gold_data
from gold_forecast.live_trading import LIVE_TRADING_PATH, run_live_trading_update
from gold_forecast.strategy_optimizer import run_optimized_strategy


def main() -> None:
    gold_ohlc = load_gold_data()
    _, leaderboard = run_optimized_strategy(gold_ohlc)
    result = run_live_trading_update(gold_ohlc, leaderboard, path=LIVE_TRADING_PATH)
    summary = result["summary"]
    print(f"Live trading ledger: {LIVE_TRADING_PATH}")
    print(f"Rows: {len(result['ledger'])}")
    print(f"Equity: {summary['Equity']:.2f}")
    print(f"Open BUY: {summary['Open BUY']} | Open SELL: {summary['Open SELL']}")
    print(f"Last update WIT: {summary['Now WIT'].strftime('%Y-%m-%d %H:%M:%S WIT')}")


if __name__ == "__main__":
    main()
