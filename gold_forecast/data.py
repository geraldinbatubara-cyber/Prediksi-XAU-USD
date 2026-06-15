from __future__ import annotations

import pandas as pd
import yfinance as yf


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
    series: dict[str, pd.Series] = {}
    for name, symbol in MARKET_SYMBOLS.items():
        data = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if data.empty or "Close" not in data.columns:
            continue
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        series[name] = close.rename(name)

    if "gold" not in series:
        raise RuntimeError("Data harga emas tidak tersedia dari Yahoo Finance.")

    frame = pd.concat(series.values(), axis=1).sort_index()
    frame = frame.ffill(limit=3).dropna(subset=["gold"])
    return frame
