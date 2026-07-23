Exit code: 0
Wall time: 2.1 seconds
Output:
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
from gold_forecast.v1_regime_classifier import run_v1_regime_classifier_lab


INPUT_DIR = PROJECT_ROOT / "data" / "intraday"
OOS_SOURCE = PROJECT_ROOT / "data" / "precomputed" / "optimizer_oos.pkl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "v1_regime_classifier.pkl.b64"
VERSION = "optimizer-v1-regime-classifier-2025-2026h1-v2"


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
    payload = run_v1_regime_classifier_lab(gold_m1, load_gold_data(), frozen)
    artifact = pickle.dumps(
        {"version": VERSION, "payload": payload}, protocol=pickle.HIGHEST_PROTOCOL
    )
    OUTPUT_PATH.write_text(base64.b64encode(artifact).decode("ascii"), encoding="ascii")

    selected = payload["methodology"]["Selected model"]
    classifier = payload["validation"].set_index("Model").loc[selected]
    economic = payload["economic"].set_index("Strategi").loc["Classifier v2 Trend Gate"]
    print(
        f"Regime Classifier v2 selesai | model={selected} | "
        f"macro_f1={classifier['Macro F1']:.3f} | balanced={classifier['Balanced accuracy']:.3f} | "
        f"growth={economic['Growth (%)']:.2f}% | PF={economic['Profit factor']:.3f} | "
        f"DD={economic['Max drawdown (%)']:.2f}% | passed={payload['decision']['Lulus seluruh kriteria']}"
    )


if __name__ == "__main__":
    main()

