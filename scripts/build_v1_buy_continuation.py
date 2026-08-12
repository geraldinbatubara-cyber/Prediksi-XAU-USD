from __future__ import annotations

import argparse
import base64
import pickle
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.v1_buy_continuation import run_v1_buy_continuation_lab
from scripts.build_v1_entry_quality_path import (
    DOWNLOAD_START,
    EXPERIMENT_END,
    _audit_monthly_coverage,
    _daily_from_m1,
)


OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "v1_buy_continuation.pkl.b64"
VERSION = "optimizer-v1-buy-specialist-bullish-continuation-2022-2026h1-v4.1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--oos-source", type=Path, required=True)
    return parser.parse_args()


def _load_history(input_dir: Path) -> pd.DataFrame:
    frames = []
    for period in pd.period_range(
        DOWNLOAD_START.to_period("M"), EXPERIMENT_END.to_period("M"), freq="M"
    ):
        path = input_dir / f"xauusd_m1_{period}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, parse_dates=["timestamp_utc"]).set_index("timestamp_utc"))
    data = pd.concat(frames).sort_index()
    return data.loc[~data.index.duplicated(keep="last")]


def main() -> None:
    args = _arguments()
    gold_m1 = _load_history(args.input_dir)
    failed = _audit_monthly_coverage(gold_m1)
    failed = failed[failed["Status"].ne("LOLOS")]
    if not failed.empty:
        raise RuntimeError(f"Audit data gagal:\n{failed.to_string(index=False)}")
    with args.oos_source.open("rb") as file:
        frozen = pickle.load(file)["payload"]
    payload = run_v1_buy_continuation_lab(
        gold_m1, _daily_from_m1(gold_m1), frozen
    )
    artifact = pickle.dumps(
        {"version": VERSION, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    OUTPUT_PATH.write_text(base64.b64encode(artifact).decode("ascii"), encoding="ascii")
    print(payload["ranking"].to_string(index=False))
    print(f"\nWinner: {payload['winner']}")


if __name__ == "__main__":
    main()
