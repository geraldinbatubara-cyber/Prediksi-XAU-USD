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

from gold_forecast.v1_sideways_specialist_v9 import run_v1_sideways_specialist_v9
from scripts.build_v1_entry_quality_path import (
    DOWNLOAD_START,
    EXPERIMENT_END,
    OOS_SOURCE,
    _audit_monthly_coverage,
)
from scripts.build_v1_entry_timing import _load_cached_history


OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "v1_sideways_specialist_v9.pkl.b64"
V8_PATH = PROJECT_ROOT / "data" / "precomputed" / "v1_sideways_specialist_v8.pkl.b64"
VERSION = "optimizer-v1-sideways-specialist-opportunity-expansion-2022-2026h1-v9"


def _load_b64_payload(path: Path):
    if not path.exists():
        return None
    return pickle.loads(base64.b64decode(path.read_bytes())).get("payload")


def _load_history_from_dir(input_dir: Path) -> pd.DataFrame:
    frames = []
    for period in pd.period_range(
        DOWNLOAD_START.to_period("M"), EXPERIMENT_END.to_period("M"), freq="M"
    ):
        path = input_dir / f"xauusd_m1_{period}.csv.gz"
        if not path.exists():
            return pd.DataFrame()
        frames.append(
            pd.read_csv(path, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
        )
    data = pd.concat(frames).sort_index()
    return data.loc[~data.index.duplicated(keep="last")]


def main() -> None:
    gold_m1 = _load_cached_history()
    if gold_m1.empty and os.getenv("GOLD_M1_CACHE_DIR"):
        gold_m1 = _load_history_from_dir(Path(os.environ["GOLD_M1_CACHE_DIR"]))
    if gold_m1.empty:
        raise RuntimeError("Cache M1 2021-10 sampai 2026-06 belum lengkap.")
    failed = _audit_monthly_coverage(gold_m1)
    failed = failed.loc[failed["Status"].ne("LOLOS")]
    if not failed.empty:
        raise RuntimeError(f"Audit data gagal:\n{failed.to_string(index=False)}")
    with OOS_SOURCE.open("rb") as file:
        frozen = pickle.load(file)["payload"]
    payload = run_v1_sideways_specialist_v9(
        gold_m1,
        frozen,
        _load_b64_payload(V8_PATH),
    )
    artifact = pickle.dumps(
        {"version": VERSION, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    OUTPUT_PATH.write_text(base64.b64encode(artifact).decode("ascii"), encoding="ascii")
    print(payload["ranking"].to_string(index=False))
    print(
        f"Winner={payload['winner']} | passed={payload['winner_passed']} | "
        f"status={payload['selection_status']}"
    )


if __name__ == "__main__":
    main()
