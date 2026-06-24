from __future__ import annotations

import time
from pathlib import Path
from tempfile import gettempdir

import pandas as pd
import yfinance as yf


YFINANCE_CACHE_DIR = Path(gettempdir()) / "prediksi-xau-usd-yfinance"
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(yf, "set_tz_cache_location"):
    yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))

MARKET_SYMBOLS = {
    "gold": "GC=F",
    "dollar": "DX-Y.NYB",
    "treasury_10y": "^TNX",
    "oil": "CL=F",
    "vix": "^VIX",
    "silver": "SI=F",
}


def load_gold_data(period: str = "10y") -> pd.DataFrame:
    """Download and normalize daily COMEX gold futures prices."""
    data = yf.download(
        "GC=F",
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError("Data harga emas tidak tersedia dari Yahoo Finance.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"Kolom data tidak lengkap: {', '.join(missing)}")

    frame = data[required].dropna(subset=["Close"]).copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame


def load_market_data(period: str = "10y") -> pd.DataFrame:
    """Download aligned close prices for gold and related global markets."""
    symbols = list(MARKET_SYMBOLS.values())
    data = pd.DataFrame()
    for attempt in range(3):
        data = yf.download(
            symbols,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )
        if not data.empty:
            break
        time.sleep(1 + attempt)

    series: dict[str, pd.Series] = {}
    if not data.empty and isinstance(data.columns, pd.MultiIndex):
        close_data = data["Close"]
        for name, symbol in MARKET_SYMBOLS.items():
            if symbol in close_data.columns and close_data[symbol].notna().any():
                series[name] = close_data[symbol].rename(name)

    if "gold" not in series:
        gold = load_gold_data(period)
        series["gold"] = gold["Close"].rename("gold")

    if "gold" not in series:
        raise RuntimeError("Data harga emas tidak tersedia dari Yahoo Finance.")

    frame = pd.concat(series.values(), axis=1).sort_index()
    frame = frame.ffill(limit=3).dropna(subset=["gold"])
    return frame
