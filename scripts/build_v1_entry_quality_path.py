from __future__ import annotations

import base64
import pickle
import sys
from datetime import timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.broker_data import apply_broker_clock_offset, load_broker_bars, load_broker_quote
from gold_forecast.v1_entry_quality_path import run_v1_entry_quality_path_lab


INPUT_DIR = PROJECT_ROOT / "data" / "intraday"
OOS_SOURCE = PROJECT_ROOT / "data" / "precomputed" / "optimizer_oos.pkl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed" / "v1_entry_quality_path.pkl.b64"
VERSION = "optimizer-v1-entry-quality-path-aware-2022-2026h1-v3"
DOWNLOAD_START = pd.Timestamp("2021-10-01")
EXPERIMENT_END = pd.Timestamp("2026-06-30 23:59:59")


def main() -> None:
    gold_m1 = _load_or_download_mt5_history()
    audit = _audit_monthly_coverage(gold_m1)
    failed = audit[audit["Status"].ne("LOLOS")]
    if not failed.empty:
        raise RuntimeError(f"Audit data gagal:\n{failed.to_string(index=False)}")

    signal_daily = _daily_from_m1(gold_m1)
    with OOS_SOURCE.open("rb") as file:
        frozen = pickle.load(file)["payload"]
    payload = run_v1_entry_quality_path_lab(gold_m1, signal_daily, frozen)
    artifact = pickle.dumps(
        {"version": VERSION, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    OUTPUT_PATH.write_text(base64.b64encode(artifact).decode("ascii"), encoding="ascii")

    methodology = payload["methodology"]
    confirmation = payload["confirmation_metrics"].iloc[0]
    economic = payload["economic"].set_index("Strategi").loc["v1 Entry Quality Path-Aware"]
    print(
        f"Path-Aware v3 selesai | model={methodology['Selected model']} | "
        f"EV>={methodology['Selected EV minimum']:.2f} | "
        f"P(TP)>={methodology['Selected TP probability minimum']:.0%} | "
        f"Brier improvement={confirmation['Brier improvement (%)']:.2f}% | "
        f"AUC={confirmation['ROC-AUC']:.3f} | growth={economic['Growth (%)']:.2f}% | "
        f"PF={economic['Profit factor']:.3f} | DD={economic['Max drawdown (%)']:.2f}% | "
        f"passed={payload['decision']['Lulus seluruh kriteria']}"
    )


def _load_or_download_mt5_history() -> pd.DataFrame:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 gagal diinisialisasi: {mt5.last_error()}")
    try:
        if not mt5.symbol_select("XAUUSD", True):
            raise RuntimeError(f"XAUUSD tidak tersedia: {mt5.last_error()}")
        tick = mt5.symbol_info_tick("XAUUSD")
        if tick is None:
            raise RuntimeError(f"Tick XAUUSD tidak tersedia: {mt5.last_error()}")
        quote = _quote_frame(tick)
        frames = []
        for period in pd.period_range(
            DOWNLOAD_START.to_period("M"), EXPERIMENT_END.to_period("M"), freq="M"
        ):
            path = INPUT_DIR / f"xauusd_m1_{period}.csv.gz"
            if path.exists():
                month = pd.read_csv(path, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
                print(f"Cached {period}: {len(month)} bars")
            else:
                start = (period.start_time - pd.Timedelta(days=1)).tz_localize(timezone.utc).to_pydatetime()
                end = ((period + 1).start_time + pd.Timedelta(days=1)).tz_localize(timezone.utc).to_pydatetime()
                rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M1, start, end)
                if rates is None or not len(rates):
                    raise RuntimeError(f"Histori M1 {period} tidak tersedia: {mt5.last_error()}")
                month = _normalize_rates(pd.DataFrame(rates), quote, period)
                if month.empty:
                    raise RuntimeError(f"Histori M1 {period} kosong setelah normalisasi.")
                INPUT_DIR.mkdir(parents=True, exist_ok=True)
                month.reset_index().to_csv(path, index=False, compression="gzip")
                print(f"Downloaded {period}: {len(month)} bars")
            frames.append(month)
    finally:
        mt5.shutdown()

    data = pd.concat(frames).sort_index()
    return data.loc[~data.index.duplicated(keep="last")]


def _quote_frame(tick) -> pd.DataFrame:
    raw = pd.DataFrame([{
        "timestamp_utc": pd.to_datetime(tick.time_msc, unit="ms", utc=True),
        "received_at_utc": pd.Timestamp.now(tz="UTC"),
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "symbol": "XAUUSD",
        "source": "MT5 DEMO",
    }])
    return load_broker_quote(raw)


def _normalize_rates(
    rates: pd.DataFrame,
    quote: pd.DataFrame,
    period: pd.Period,
) -> pd.DataFrame:
    bars = rates.rename(columns={"time": "timestamp_utc", "spread": "spread_points"})
    bars["timestamp_utc"] = pd.to_datetime(bars["timestamp_utc"], unit="s", utc=True)
    bars["symbol"] = "XAUUSD"
    bars["source"] = "MT5 DEMO"
    clean = load_broker_bars(bars)
    clean, _ = apply_broker_clock_offset(clean, quote)
    indexed = clean.set_index("timestamp_utc")[["open", "high", "low", "close", "spread_points"]]
    indexed.index = indexed.index.tz_convert("UTC").tz_localize(None)
    indexed = indexed.loc[(indexed.index >= period.start_time) & (indexed.index <= period.end_time)]
    return indexed.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "spread_points": "SpreadPoints",
    })


def _daily_from_m1(data: pd.DataFrame) -> pd.DataFrame:
    daily = data.resample("1D").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna(subset=["Close"])
    daily["Volume"] = 0.0
    return daily


def _audit_monthly_coverage(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in pd.period_range("2022-01", "2026-06", freq="M"):
        month = data.loc[
            (data.index >= period.start_time) & (data.index <= period.end_time)
        ]
        invalid = int(
            (
                (month["Low"] > month[["Open", "Close"]].min(axis=1))
                | (month["High"] < month[["Open", "Close"]].max(axis=1))
                | (month["High"] < month["Low"])
            ).sum()
        ) if not month.empty else 0
        rows.append({
            "Bulan": str(period),
            "Bars": len(month),
            "Invalid OHLC": invalid,
            "Status": "LOLOS" if len(month) >= 20_000 and invalid == 0 else "BELUM",
        })
    audit = pd.DataFrame(rows)
    print(audit.to_string(index=False))
    return audit


if __name__ == "__main__":
    main()
