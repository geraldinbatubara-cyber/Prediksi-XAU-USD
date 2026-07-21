from __future__ import annotations

import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.data import load_gold_data
from gold_forecast.martingale import run_martingale_v1


OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "martingale_v1.pkl"
VERSION = "martingale-v1-daily-2025q1-2026q2"


def main() -> None:
    result, leaderboard = run_martingale_v1(load_gold_data())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as file:
        pickle.dump(
            {"version": VERSION, "payload": (result, leaderboard)},
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    summary = result.summary
    print(
        f"Saved {OUTPUT_PATH} | equity={summary['Equity akhir']:.2f} | "
        f"growth={summary['Growth total']:+.2f}% | baskets={summary['Jumlah basket']:.0f} | "
        f"risk_pass={bool(leaderboard.iloc[0]['Lolos batas risiko'])}"
    )


if __name__ == "__main__":
    main()
