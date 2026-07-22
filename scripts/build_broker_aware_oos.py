from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.m1_backtest import run_intraday_optimization


HISTORY_DIR = PROJECT_ROOT / "data" / "intraday"
OOS_SOURCE = PROJECT_ROOT / "data" / "precomputed" / "optimizer_oos.pkl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "broker_aware_oos.pkl"
VERSION = "optimizer-v1-v10-broker-aware-train2025-oos2026h1"
START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2025-12-31 23:59:59")
TEST_START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-06-30 23:59:59")


def main() -> None:
    gold_m1 = _load_m1()
    gold_daily = _daily_from_m1(gold_m1)
    frozen_daily = _load_frozen_daily_parameters()
    payload = {}
    for variant in ("v1", "v10"):
        result, leaderboard = run_intraday_optimization(
            gold_m1,
            gold_daily,
            variant=variant,
            requested_start=START,
            requested_end=END,
            train_start=START,
            train_end=TRAIN_END,
            test_start=TEST_START,
            test_end=END,
            daily_params=frozen_daily[variant],
        )
        result = _attach_broker_costs(result, gold_m1)
        payload[variant] = (_compact(result), leaderboard)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as file:
        pickle.dump({"version": VERSION, "payload": payload}, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Saved {OUTPUT_PATH} | v1={payload['v1'][0].summary['Growth total']:+.2f}% | "
        f"v10={payload['v10'][0].summary['Growth total']:+.2f}%"
    )


def _load_m1() -> pd.DataFrame:
    frames = []
    for period in pd.period_range(START.to_period("M"), END.to_period("M"), freq="M"):
        path = HISTORY_DIR / f"xauusd_m1_{period}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Histori broker belum lengkap: {path.name} tidak ditemukan.")
        frame = pd.read_csv(path, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
        frames.append(frame)
    data = pd.concat(frames).sort_index()
    data = data.loc[~data.index.duplicated(keep="last")]
    return data.loc[(data.index >= START) & (data.index <= END)]


def _daily_from_m1(data: pd.DataFrame) -> pd.DataFrame:
    daily = data.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    return daily.dropna().astype(float)


def _load_frozen_daily_parameters() -> dict[str, dict[str, object]]:
    saved = pickle.load(OOS_SOURCE.open("rb"))
    payload = saved["payload"]
    result = {}
    for variant in ("v1", "v10"):
        leaderboard = payload[variant][1]
        best = leaderboard.iloc[0]
        result[variant] = {
            "Mode": best["Mode"],
            "Fast MA": best["Fast MA"],
            "Slow MA": best["Slow MA"],
            "Momentum hari": best["Momentum hari"],
            "Threshold entry (%)": best["Threshold entry (%)"],
        }
    return result


def _attach_broker_costs(result, data: pd.DataFrame):
    trades = result.trades.copy()
    summary = result.summary
    spread_points = pd.to_numeric(data.loc[(data.index >= TEST_START) & (data.index <= END), "SpreadPoints"], errors="coerce")
    spread_costs = []
    slippage_costs = []
    if not trades.empty:
        for _, trade in trades.iterrows():
            timestamp = pd.Timestamp(trade["Tanggal entry"] if trade["Arah"] == "BUY" else trade["Tanggal tutup"])
            location = data.index.get_indexer([timestamp], method="nearest")[0]
            points = float(data.iloc[location]["SpreadPoints"])
            units = float(trade["Lot"]) * 100.0
            spread_costs.append(points * 0.01 * units)
            slippage_costs.append(2 * 2 * 0.01 * units)
        trades["Biaya spread estimasi"] = spread_costs
        trades["Biaya slippage estimasi"] = slippage_costs
    summary.update(
        {
            "Biaya spread estimasi": float(sum(spread_costs)),
            "Biaya slippage estimasi": float(sum(slippage_costs)),
            "Spread rata-rata points": float(spread_points.mean()),
            "Spread median points": float(spread_points.median()),
            "Spread p95 points": float(spread_points.quantile(0.95)),
            "Spread maksimum points": float(spread_points.max()),
            "Candle M1 train": float(len(data.loc[(data.index >= START) & (data.index <= TRAIN_END)])),
            "Candle M1 OOS": float(len(data.loc[(data.index >= TEST_START) & (data.index <= END)])),
            "Data M1 mulai": data.index.min(),
            "Data M1 akhir": data.index.max(),
            "Cakupan lengkap": set(data.index.to_period("M").unique()) == set(pd.period_range("2025-01", "2026-06", freq="M")),
        }
    )
    return replace(result, trades=trades)


def _compact(result):
    curve = result.equity_curve
    if len(curve) > 6000:
        important = pd.concat([curve.loc[[curve["Equity"].idxmin()]], curve.loc[[curve["Equity"].idxmax()]], curve.tail(1)])
        curve = pd.concat([curve.iloc[::30], important]).sort_index()
        curve = curve.loc[~curve.index.duplicated(keep="last")]
    return replace(result, equity_curve=curve, trades=result.trades.tail(2000).copy())


if __name__ == "__main__":
    main()
