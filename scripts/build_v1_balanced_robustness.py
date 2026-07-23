from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.data import load_gold_data
from gold_forecast.v1_balanced_robustness import run_v1_balanced_robustness_lab


INPUT_DIR = PROJECT_ROOT / "data" / "intraday"
OOS_SOURCE = PROJECT_ROOT / "data" / "precomputed" / "optimizer_oos.pkl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "v1_balanced_robustness.pkl.b64"
VERSION = "optimizer-v1-balanced-entry-robustness-2025-2026h1-v1"


def main() -> None:
    frames = []
    for path in sorted(INPUT_DIR.glob("xauusd_m1_*.csv.gz")):
        period = path.stem.replace("xauusd_m1_", "").replace(".csv", "")
        if "2025-01" <= period <= "2026-06":
            frames.append(pd.read_csv(path, parse_dates=["timestamp_utc"]).set_index("timestamp_utc"))
    if not frames:
        raise RuntimeError("Dataset M1 bulanan 2025-01 sampai 2026-06 tidak ditemukan.")
    gold_m1 = pd.concat(frames).sort_index()
    with OOS_SOURCE.open("rb") as file:
        frozen = pickle.load(file)["payload"]
    payload = run_v1_balanced_robustness_lab(gold_m1, load_gold_data(), frozen)
    artifact = pickle.dumps(
        {"version": VERSION, "payload": payload}, protocol=pickle.HIGHEST_PROTOCOL
    )
    OUTPUT_PATH.write_text(base64.b64encode(artifact).decode("ascii"), encoding="ascii")
    summary = payload["stability"]
    print(
        f"Balanced Robustness selesai | status={summary['Robustness status']} | "
        f"growth_positive={summary['Growth positif (%)']:.1f}% | "
        f"pf_pass={summary['Profit factor >= 1.30 (%)']:.1f}% | "
        f"dd_pass={summary['Drawdown <= 10% (%)']:.1f}%"
    )


if __name__ == "__main__":
    main()
