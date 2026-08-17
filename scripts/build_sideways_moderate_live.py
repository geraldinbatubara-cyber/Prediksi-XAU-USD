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

from gold_forecast.exact_broker_oos import _prepare_m1
from gold_forecast.sideways_moderate_live import build_moderate_regime_live_bundle
from gold_forecast.v1_sell_specialist import DEVELOPMENT_END, DEVELOPMENT_START
from scripts.build_v1_entry_timing import _load_cached_history
from scripts.build_v1_sideways_specialist_v9 import _load_history_from_dir


OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "sideways_moderate_live.pkl.b64"
V9_PATH = PROJECT_ROOT / "data" / "precomputed" / "v1_sideways_specialist_v9.pkl.b64"
VERSION = "sideways-v9-moderate-regime-live-inference-2026-08-17-v1"


def main() -> None:
    historical = _load_cached_history()
    if historical.empty and os.getenv("GOLD_M1_CACHE_DIR"):
        historical = _load_history_from_dir(Path(os.environ["GOLD_M1_CACHE_DIR"]))
    if historical.empty:
        raise RuntimeError("Cache M1 historis belum tersedia.")
    data = _prepare_m1(historical)
    saved = pickle.loads(base64.b64decode(V9_PATH.read_bytes()))
    payload = saved["payload"]
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    bundle = build_moderate_regime_live_bundle(
        data, payload["expansion_thresholds"], spread_limit
    )
    bundle["generated_at_utc"] = pd.Timestamp.now(tz="UTC")
    artifact = pickle.dumps(
        {"version": VERSION, "payload": bundle},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    OUTPUT_PATH.write_text(base64.b64encode(artifact).decode("ascii"), encoding="ascii")
    print(f"Moderate Regime live bundle selesai | size={OUTPUT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
