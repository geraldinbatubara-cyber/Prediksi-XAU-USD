from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.data import GOLD_CACHE_PATH, MARKET_CACHE_PATH, refresh_market_cache


def main() -> None:
    gold, market = refresh_market_cache()
    print(
        f"Gold cache: {GOLD_CACHE_PATH} | rows={len(gold)} | "
        f"range={gold.index.min().date()} to {gold.index.max().date()}"
    )
    print(
        f"Market cache: {MARKET_CACHE_PATH} | rows={len(market)} | "
        f"range={market.index.min().date()} to {market.index.max().date()}"
    )


if __name__ == "__main__":
    main()
