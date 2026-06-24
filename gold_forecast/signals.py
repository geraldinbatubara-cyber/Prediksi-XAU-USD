from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FACTOR_LABELS = {
    "dollar": "DXY",
    "treasury_10y": "Yield Treasury 10Y",
    "oil": "Minyak WTI",
    "vix": "VIX",
    "silver": "Perak",
}


@dataclass
class TradingSignal:
    label: str
    confidence: float
    expected_change: float
    expected_change_pct: float
    rationale: list[str]
    drivers: pd.DataFrame


def _direction_label(expected_change_pct: float, confidence: float) -> str:
    if confidence < 45 or abs(expected_change_pct) < 0.15:
        return "Netral"
    if expected_change_pct > 0:
        return "Bullish"
    return "Bearish"


def _factor_bias(name: str, change_pct: float) -> str:
    if name in {"dollar", "treasury_10y", "vix"}:
        return "Bearish" if change_pct > 0 else "Bullish"
    if name in {"oil", "silver"}:
        return "Bullish" if change_pct > 0 else "Bearish"
    return "Netral"


def _factor_note(name: str, change_pct: float) -> str:
    direction = "naik" if change_pct > 0 else "turun"
    label = FACTOR_LABELS.get(name, name)
    bias = _factor_bias(name, change_pct)
    return f"{label} {direction} {abs(change_pct):.2f}% dan memberi bias {bias.lower()}."


def build_signal(market: pd.DataFrame, forecast: pd.DataFrame) -> TradingSignal:
    gold = market["gold"].dropna()
    latest = float(gold.iloc[-1])
    tomorrow = forecast.iloc[0]
    estimate = float(tomorrow["Estimasi"])
    lower = float(tomorrow["Batas bawah"])
    upper = float(tomorrow["Batas atas"])
    expected_change = estimate - latest
    expected_change_pct = expected_change / latest * 100

    daily_volatility_pct = float(gold.pct_change().tail(60).std() * 100)
    if not np.isfinite(daily_volatility_pct) or daily_volatility_pct <= 0:
        daily_volatility_pct = 1.0

    signal_strength = min(1.0, abs(expected_change_pct) / (daily_volatility_pct * 1.5))
    interval_width_pct = (upper - lower) / latest * 100
    interval_penalty = min(0.55, interval_width_pct / 12)
    confidence = max(20.0, min(85.0, 35 + signal_strength * 65 - interval_penalty * 100))
    label = _direction_label(expected_change_pct, confidence)

    driver_rows: list[dict[str, str | float]] = []
    for name in FACTOR_LABELS:
        if name not in market.columns:
            continue
        values = market[name].dropna()
        if len(values) < 6:
            continue
        change_pct = float((values.iloc[-1] / values.iloc[-6] - 1) * 100)
        if not np.isfinite(change_pct):
            continue
        driver_rows.append(
            {
                "Faktor": FACTOR_LABELS[name],
                "Perubahan 5 hari": change_pct,
                "Bias ke emas": _factor_bias(name, change_pct),
                "Catatan": _factor_note(name, change_pct),
            }
        )

    drivers = pd.DataFrame(driver_rows)
    rationale = [
        f"Estimasi besok bergerak {expected_change_pct:+.2f}% dari harga terakhir.",
        f"Volatilitas 60 hari sekitar {daily_volatility_pct:.2f}% per hari.",
        f"Lebar interval estimasi T+1 sekitar {interval_width_pct:.2f}% dari harga terakhir.",
    ]
    if not drivers.empty:
        top_driver = drivers.reindex(
            drivers["Perubahan 5 hari"].abs().sort_values(ascending=False).index
        ).iloc[0]
        rationale.append(str(top_driver["Catatan"]))

    return TradingSignal(
        label=label,
        confidence=confidence,
        expected_change=expected_change,
        expected_change_pct=expected_change_pct,
        rationale=rationale,
        drivers=drivers,
    )
