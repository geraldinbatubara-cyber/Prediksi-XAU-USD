from __future__ import annotations

import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.data import load_gold_data
from gold_forecast.optimizer_oos import run_optimizer_oos


OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "optimizer_oos.pkl"
VERSION = "optimizer-v1-only-train2025-oos2026h1"


def main() -> None:
    payload = run_optimizer_oos(load_gold_data())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as file:
        pickle.dump({"version": VERSION, "payload": payload}, file, protocol=pickle.HIGHEST_PROTOCOL)
    v1 = payload["v1"][2].summary
    print(f"Saved {OUTPUT_PATH} | v1 OOS={v1['Growth total']:+.2f}%")


if __name__ == "__main__":
    main()
