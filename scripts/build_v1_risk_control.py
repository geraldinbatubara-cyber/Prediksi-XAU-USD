from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.data import load_gold_data
from gold_forecast.v1_risk_control import run_v1_risk_control_lab


HISTORY_DIR = PROJECT_ROOT / "data" / "intraday"
OOS_SOURCE = PROJECT_ROOT / "data" / "precomputed" / "optimizer_oos.pkl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "v1_risk_control.pkl"
VERSION = "optimizer-v1-risk-control-lab-2025-2026h1-v1"


def main() -> None:
    frames = []
    for period in pd.period_range("2025-01", "2026-06", freq="M"):
        path = HISTORY_DIR / f"xauusd_m1_{period}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Histori broker belum lengkap: {path.name}")
        frames.append(pd.read_csv(path, parse_dates=["timestamp_utc"]).set_index("timestamp_utc"))
    gold_m1 = pd.concat(frames).sort_index()
    with OOS_SOURCE.open("rb") as file:
        frozen = pickle.load(file)["payload"]
    payload = run_v1_risk_control_lab(gold_m1, load_gold_data(), frozen)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as file:
        pickle.dump({"version": VERSION, "payload": payload}, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Saved {OUTPUT_PATH} | winner={payload['winner_name']} | "
        f"status={payload['winner_status']} | candidates={len(payload['development'])}"
    )


if __name__ == "__main__":
    main()
