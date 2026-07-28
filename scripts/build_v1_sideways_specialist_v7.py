from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.v1_sideways_specialist_v7 import (
    run_v1_sideways_specialist_v7,
)
from scripts.build_v1_entry_quality_path import OOS_SOURCE, _audit_monthly_coverage
from scripts.build_v1_entry_timing import _load_cached_history


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "precomputed"
    / "v1_sideways_specialist_v7.pkl.b64"
)
DEFENSE_PATH = (
    PROJECT_ROOT / "data" / "precomputed" / "v1_sideways_defense.pkl.b64"
)
V6_PATH = (
    PROJECT_ROOT
    / "data"
    / "precomputed"
    / "v1_sideways_specialist_v6.pkl.b64"
)
VERSION = "optimizer-v1-sideways-specialist-mtf-native-persistence-2022-2026h1-v7"


def _load_b64_payload(path: Path):
    if not path.exists():
        return None
    return pickle.loads(base64.b64decode(path.read_bytes())).get("payload")


def main() -> None:
    gold_m1 = _load_cached_history()
    if gold_m1.empty:
        raise RuntimeError("Cache M1 2021-10 sampai 2026-06 belum lengkap.")
    failed = _audit_monthly_coverage(gold_m1)
    failed = failed.loc[failed["Status"].ne("LOLOS")]
    if not failed.empty:
        raise RuntimeError(f"Audit data gagal:\n{failed.to_string(index=False)}")
    with OOS_SOURCE.open("rb") as file:
        frozen = pickle.load(file)["payload"]
    payload = run_v1_sideways_specialist_v7(
        gold_m1,
        frozen,
        _load_b64_payload(DEFENSE_PATH),
        _load_b64_payload(V6_PATH),
    )
    artifact = pickle.dumps(
        {"version": VERSION, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    OUTPUT_PATH.write_text(
        base64.b64encode(artifact).decode("ascii"),
        encoding="ascii",
    )
    print(payload["ranking"].to_string(index=False))
    print(
        f"Winner={payload['winner']} | passed={payload['winner_passed']} | "
        f"classifier={payload['methodology']['Selected H1 regime classifier']}"
    )


if __name__ == "__main__":
    main()
