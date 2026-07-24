from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.v1_directional_specialization import (
    run_v1_directional_specialization_lab,
)
from scripts.build_v1_entry_quality_path import (
    OOS_SOURCE,
    _audit_monthly_coverage,
    _daily_from_m1,
)
from scripts.build_v1_entry_timing import _load_cached_history


OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "precomputed" / "v1_directional_specialization.pkl.b64"
)
VERSION = "optimizer-v1-directional-specialization-2022-2026h1-v4"


def main() -> None:
    gold_m1 = _load_cached_history()
    if gold_m1.empty:
        raise RuntimeError("Cache M1 2021-10 sampai 2026-06 belum lengkap.")
    audit = _audit_monthly_coverage(gold_m1)
    failed = audit[audit["Status"].ne("LOLOS")]
    if not failed.empty:
        raise RuntimeError(f"Audit data gagal:\n{failed.to_string(index=False)}")

    with OOS_SOURCE.open("rb") as file:
        frozen = pickle.load(file)["payload"]
    payload = run_v1_directional_specialization_lab(
        gold_m1,
        _daily_from_m1(gold_m1),
        frozen,
    )
    artifact = pickle.dumps(
        {"version": VERSION, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    OUTPUT_PATH.write_text(
        base64.b64encode(artifact).decode("ascii"),
        encoding="ascii",
    )

    long_winner = payload["long_ranking"].iloc[0]
    symmetric_winner = payload["symmetric_ranking"].iloc[0]
    print(
        f"Directional Specialization selesai | long={long_winner['Kandidat']} "
        f"({long_winner['Kriteria lolos']}/9, passed={long_winner['Lulus']}) | "
        f"symmetric={symmetric_winner['Kandidat']} "
        f"({symmetric_winner['Kriteria lolos']}/13, passed={symmetric_winner['Lulus']})"
    )
    print("\nLONG TRACK")
    print(payload["long_ranking"].to_string(index=False))
    print("\nSYMMETRIC TRACK")
    print(payload["symmetric_ranking"].to_string(index=False))


if __name__ == "__main__":
    main()
