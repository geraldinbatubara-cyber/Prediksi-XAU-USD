from __future__ import annotations

import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.dashboard_snapshot import (
    DASHBOARD_SNAPSHOT_PATH,
    V10_PARAMS_PATH,
    build_dashboard_snapshot,
    load_v10_params,
    save_dashboard_snapshot,
    save_v10_params,
)
from gold_forecast.data import _download_market_data, load_market_data
from gold_forecast.strategy_optimizer import run_optimized_strategy_v10


def main() -> None:
    started_at = time.perf_counter()
    market = load_market_data()
    if len(market) < 650:
        print("Cache 2025+ belum cukup untuk training; mengambil riwayat 5 tahun hanya untuk snapshot model.")
        market = _download_market_data("5y")
    v10_leaderboard = load_v10_params()

    if v10_leaderboard.empty:
        print("Parameter v10 belum tersedia; menjalankan optimasi awal satu kali.")
        from gold_forecast.data import load_gold_data

        _, v10_leaderboard = run_optimized_strategy_v10(load_gold_data())
        save_v10_params(v10_leaderboard)

    snapshot = build_dashboard_snapshot(market, v10_leaderboard)
    save_dashboard_snapshot(snapshot)
    elapsed = time.perf_counter() - started_at
    print(
        f"Dashboard snapshot: {DASHBOARD_SNAPSHOT_PATH} | "
        f"v10 params: {V10_PARAMS_PATH} | rows={len(market)} | elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()

