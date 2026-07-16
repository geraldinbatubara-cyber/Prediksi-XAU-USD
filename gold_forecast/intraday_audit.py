from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO

import numpy as np
import pandas as pd


REQUIRED_OHLC_COLUMNS = ("time", "open", "high", "low", "close")
OPTIONAL_COLUMNS = ("volume",)


@dataclass(frozen=True)
class IntradayAuditResult:
    data: pd.DataFrame
    metrics: dict[str, object]
    issues: pd.DataFrame
    daily_ohlc: pd.DataFrame


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]

    aliases = {
        "datetime": "time",
        "date": "time",
        "timestamp": "time",
        "time_utc": "time",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "vol": "volume",
        "tick_volume": "volume",
    }
    return normalized.rename(columns={source: target for source, target in aliases.items() if source in normalized.columns})


def load_intraday_csv(source) -> pd.DataFrame:
    if isinstance(source, bytes):
        raw = BytesIO(source)
    elif isinstance(source, str):
        raw = StringIO(source) if "\n" in source else source
    else:
        raw = source

    frame = pd.read_csv(raw)
    frame = _normalize_columns(frame)
    missing = [column for column in REQUIRED_OHLC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")

    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    for column in REQUIRED_OHLC_COLUMNS[1:] + OPTIONAL_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return frame


def audit_intraday_data(frame: pd.DataFrame) -> IntradayAuditResult:
    if frame.empty:
        return IntradayAuditResult(
            data=frame,
            metrics={"Status": "Kosong"},
            issues=pd.DataFrame(columns=["Kategori", "Jumlah", "Catatan"]),
            daily_ohlc=pd.DataFrame(),
        )

    data = frame.copy()
    duplicate_count = int(data["time"].duplicated().sum())
    data = data.drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)
    data = data.set_index("time")

    numeric_missing = data[["open", "high", "low", "close"]].isna().sum()
    invalid_high_low = int((data["high"] < data["low"]).sum())
    invalid_open_range = int(((data["open"] > data["high"]) | (data["open"] < data["low"])).sum())
    invalid_close_range = int(((data["close"] > data["high"]) | (data["close"] < data["low"])).sum())

    minute_gap = data.index.to_series().diff().dt.total_seconds().div(60)
    expected_minutes = 0
    missing_minutes_estimate = 0
    if len(data.index) >= 2:
        expected_minutes = int((data.index.max() - data.index.min()).total_seconds() // 60) + 1
        missing_minutes_estimate = max(expected_minutes - len(data), 0)
    large_gaps = minute_gap[minute_gap > 1].dropna()

    returns = data["close"].pct_change() * 100
    outlier_cutoff = max(float(returns.abs().quantile(0.999)) if returns.notna().any() else 0.0, 1.0)
    outliers = returns.abs() > outlier_cutoff

    daily = data.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum") if "volume" in data.columns else ("close", "count"),
    )
    daily = daily.dropna(subset=["open", "high", "low", "close"])

    issues = pd.DataFrame(
        [
            {
                "Kategori": "Duplikat timestamp",
                "Jumlah": duplicate_count,
                "Catatan": "Baris dengan waktu sama. Audit memakai baris terakhir.",
            },
            {
                "Kategori": "Estimasi missing menit",
                "Jumlah": int(missing_minutes_estimate),
                "Catatan": "Estimasi kasar dari rentang waktu kontinu UTC; market close/weekend bisa ikut terhitung.",
            },
            {
                "Kategori": "Gap > 1 menit",
                "Jumlah": int(len(large_gaps)),
                "Catatan": "Cek apakah gap berasal dari libur market atau data hilang.",
            },
            {
                "Kategori": "High < Low",
                "Jumlah": invalid_high_low,
                "Catatan": "Candle tidak valid.",
            },
            {
                "Kategori": "Open di luar High-Low",
                "Jumlah": invalid_open_range,
                "Catatan": "Open harus berada dalam rentang high-low.",
            },
            {
                "Kategori": "Close di luar High-Low",
                "Jumlah": invalid_close_range,
                "Catatan": "Close harus berada dalam rentang high-low.",
            },
            {
                "Kategori": "OHLC kosong",
                "Jumlah": int(numeric_missing.sum()),
                "Catatan": ", ".join(f"{key}: {value}" for key, value in numeric_missing.items()),
            },
            {
                "Kategori": "Outlier return 1 menit",
                "Jumlah": int(outliers.sum()),
                "Catatan": f"Return absolut di atas ambang audit {outlier_cutoff:.2f}%.",
            },
        ]
    )

    metrics = {
        "Status": "Perlu review" if int(issues["Jumlah"].sum()) else "Tidak ada isu besar terdeteksi",
        "Jumlah baris": int(len(data)),
        "Periode awal UTC": data.index.min(),
        "Periode akhir UTC": data.index.max(),
        "Jumlah hari agregasi": int(len(daily)),
        "Duplikat timestamp": duplicate_count,
        "Gap > 1 menit": int(len(large_gaps)),
        "Estimasi missing menit": int(missing_minutes_estimate),
        "Return 1m median": float(returns.median()) if returns.notna().any() else np.nan,
        "Return 1m p99 absolut": float(returns.abs().quantile(0.99)) if returns.notna().any() else np.nan,
        "Timezone asumsi": "UTC dari parser pandas; tampilkan ke WIT di dashboard jika dipakai live.",
    }
    return IntradayAuditResult(data=data, metrics=metrics, issues=issues, daily_ohlc=daily)
