from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.v1_fixed_delay_quality import (
    run_v1_fixed_delay_quality_lab,
)
from scripts.build_v1_entry_quality_path import (
    OOS_SOURCE,
    _audit_monthly_coverage,
    _daily_from_m1,
)
from scripts.build_v1_entry_timing import _load_cached_history


OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "precomputed" / "v1_fixed_delay_quality.pkl.b64"
)
VERSION = "optimizer-v1-fixed-delay-quality-guard-2022-2026h1-v1"


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
    payload = run_v1_fixed_delay_quality_lab(
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

    winner = payload["ranking"].iloc[0]
    print(
        f"Fixed Delay Quality Guard selesai | winner={winner['Kandidat']} | "
        f"dev growth={winner['Growth development (%)']:.2f}% | "
        f"dev PF={winner['PF development']:.3f} | "
        f"dev DD={winner['DD development (%)']:.2f}% | "
        f"criteria={winner['Kriteria lolos']}/7 | passed={winner['Lulus']}"
    )
    print(payload["ranking"].to_string(index=False))


if __name__ == "__main__":
    main()
