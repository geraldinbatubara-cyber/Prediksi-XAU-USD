from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.v1_sideways_specialist_v3 import (
    run_v1_sideways_specialist_v3_lab,
)
from scripts.build_v1_entry_quality_path import OOS_SOURCE, _audit_monthly_coverage
from scripts.build_v1_entry_timing import _load_cached_history


OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "precomputed" / "v1_sideways_specialist_v3.pkl.b64"
)
V2_PATH = PROJECT_ROOT / "data" / "precomputed" / "v1_sideways_specialist_v2.pkl.b64"
VERSION = "optimizer-v1-sideways-specialist-dynamic-hazard-2022-2026h1-v3"


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
    v2_payload = None
    if V2_PATH.exists():
        artifact = pickle.loads(base64.b64decode(V2_PATH.read_bytes()))
        v2_payload = artifact.get("payload")
    payload = run_v1_sideways_specialist_v3_lab(
        gold_m1, frozen, v2_payload
    )
    artifact = pickle.dumps(
        {"version": VERSION, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    OUTPUT_PATH.write_text(
        base64.b64encode(artifact).decode("ascii"),
        encoding="ascii",
    )
    top = payload["ranking"].iloc[0]
    print(
        f"Sideways Specialist v3 selesai | selection={payload['selection_status']} | "
        f"top={top['Kandidat']} | criteria={top['Kriteria lolos']}/10 | "
        f"passed={payload['winner_passed']}"
    )
    print(payload["ranking"].to_string(index=False))


if __name__ == "__main__":
    main()
