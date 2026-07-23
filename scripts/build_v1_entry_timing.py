from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.v1_entry_timing import run_v1_entry_timing_lab
from scripts.build_v1_entry_quality_path import (
    DOWNLOAD_START,
    EXPERIMENT_END,
    INPUT_DIR,
    OOS_SOURCE,
    _audit_monthly_coverage,
    _daily_from_m1,
    _load_or_download_mt5_history,
)


OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "v1_entry_timing.pkl.b64"
VERSION = "optimizer-v1-entry-timing-micro-confirmation-2022-2026h1-v1"


def main() -> None:
    gold_m1 = _load_cached_history()
    if gold_m1.empty:
        gold_m1 = _load_or_download_mt5_history()
    audit = _audit_monthly_coverage(gold_m1)
    failed = audit[audit["Status"].ne("LOLOS")]
    if not failed.empty:
        raise RuntimeError(f"Audit data gagal:\n{failed.to_string(index=False)}")

    with OOS_SOURCE.open("rb") as file:
        frozen = pickle.load(file)["payload"]
    payload = run_v1_entry_timing_lab(
        gold_m1, _daily_from_m1(gold_m1), frozen
    )
    artifact = pickle.dumps(
        {"version": VERSION, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    OUTPUT_PATH.write_text(
        base64.b64encode(artifact).decode("ascii"),
        encoding="ascii",
    )

    methodology = payload["methodology"]
    economic = (
        payload["economic"]
        .set_index("Strategi")
        .loc["v1 Micro Confirmation"]
    )
    print(
        f"Micro Confirmation selesai | rule={methodology['Selected rule']} | "
        f"growth={economic['Growth (%)']:.2f}% | PF={economic['Profit factor']:.3f} | "
        f"DD={economic['Max drawdown (%)']:.2f}% | tx={economic['Transaksi']:.0f} | "
        f"passed={payload['decision']['Lulus seluruh kriteria']}"
    )


def _load_cached_history() -> pd.DataFrame:
    frames = []
    for period in pd.period_range(
        DOWNLOAD_START.to_period("M"), EXPERIMENT_END.to_period("M"), freq="M"
    ):
        path = INPUT_DIR / f"xauusd_m1_{period}.csv.gz"
        if not path.exists():
            return pd.DataFrame()
        frames.append(
            pd.read_csv(path, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
        )
    data = pd.concat(frames).sort_index()
    return data.loc[~data.index.duplicated(keep="last")]


if __name__ == "__main__":
    main()
