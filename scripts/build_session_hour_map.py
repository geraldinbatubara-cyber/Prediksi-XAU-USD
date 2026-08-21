from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.session_hour_map import build_session_hour_map
from gold_forecast.post_entry_audit import build_post_entry_audit


START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-06-30")
OUTPUT_DIR = PROJECT_ROOT / "data" / "precomputed"


def _load_months(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    sources = []
    for period in pd.period_range(START.to_period("M"), END.to_period("M"), freq="M"):
        path = input_dir / f"xauusd_m1_{period}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Dataset bulanan tidak ditemukan: {path}")
        frames.append(pd.read_csv(path, compression="gzip"))
        sources.append(path.name)
    return pd.concat(frames, ignore_index=True), sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Bangun pemetaan jam OHLC sesi XAUUSD dari candle M1.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "intraday")
    args = parser.parse_args()

    bars, sources = _load_months(args.input_dir)
    result = build_session_hour_map(bars, START, END)
    post_entry = build_post_entry_audit(bars, START, END)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(OUTPUT_DIR / "xauusd_session_hour_map.csv", index=False)
    result.hourly_frequency.to_csv(OUTPUT_DIR / "xauusd_session_hour_frequency.csv", index=False)
    result.monthly_frequency.to_csv(OUTPUT_DIR / "xauusd_session_hour_monthly.csv", index=False)
    metadata = {
        **result.metadata,
        "source": "MT5 DEMO XAUUSD M1 cached history",
        "source_files": sources,
        "timezone_input": "UTC (timestamp_utc yang telah dikoreksi dari waktu server MT5)",
        "timezone_output": "Asia/Jayapura (WIT, UTC+9)",
        "interpolation": "Tidak digunakan",
    }
    (OUTPUT_DIR / "xauusd_session_hour_audit.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    post_entry.paths.to_csv(OUTPUT_DIR / "xauusd_post_0800_paths.csv", index=False)
    post_entry.summary.to_csv(OUTPUT_DIR / "xauusd_post_0800_summary.csv", index=False)
    (OUTPUT_DIR / "xauusd_post_0800_audit.json").write_text(
        json.dumps(post_entry.metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(json.dumps(post_entry.metadata, indent=2))


if __name__ == "__main__":
    main()
