from __future__ import annotations

import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.data import load_gold_data
from gold_forecast.martingale_v3 import run_martingale_v3


OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "martingale_v3.pkl"
VERSION = "martingale-v3-v10-recovery-train2025-oos2026h1"


def main() -> None:
    full_result, leaderboard, train_result, oos_result = run_martingale_v3(load_gold_data())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as file:
        pickle.dump(
            {
                "version": VERSION,
                "payload": (full_result, leaderboard, train_result, oos_result),
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    summary = full_result.summary
    print(
        f"Saved {OUTPUT_PATH} | full={summary['Growth total']:+.2f}% | "
        f"oos={summary['OOS growth (%)']:+.2f}% | oos_baskets={summary['OOS jumlah basket']:.0f} | "
        f"status={summary['Status kelayakan']}"
    )


if __name__ == "__main__":
    main()
