from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import SLIPPAGE_POINTS, _compact_curve, _prepare_m1
from gold_forecast.v1_risk_control import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    MAX_DRAWDOWN_PCT,
    PROFIT_FACTOR_TARGET,
    VALIDATION_END,
    VALIDATION_START,
    RiskControlConfig,
    _entry_signals_for_period,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_robustness import _monte_carlo, _monthly_summary, _profit_factor
from gold_forecast.v1_signal_quality import (
    SignalQualityConfig,
    _entry_features,
    _positive_quarters,
    _select_signals,
)


MA_PAIRS = ((8, 24), (10, 30), (12, 36))
WAIT_HOURS = (1, 2, 3)
CONVICTION_MULTIPLIERS = (1.00, 1.05, 1.10)
CENTER_PARAMETERS = (10, 30, 2, 1.05)


@dataclass(frozen=True)
class RobustnessConfig:
    fast_ma: int
    slow_ma: int
    wait_hours: int
    conviction_multiplier: float

    @property
    def name(self) -> str:
        conviction = int(round(self.conviction_multiplier * 100))
        return f"H1 {self.fast_ma}/{self.slow_ma} | Wait {self.wait_hours}h | Conv {conviction}%"


def run_v1_balanced_robustness_lab(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = leaderboard.iloc[0].to_dict()
    development = data.loc[(data.index >= DEVELOPMENT_START) & (data.index <= DEVELOPMENT_END)]
    validation = data.loc[(data.index >= VALIDATION_START) & (data.index <= VALIDATION_END)]
    if development.empty or validation.empty:
        raise ValueError("Dataset M1 development 2025 atau validation 2026H1 belum lengkap.")

    raw_development = _entry_signals_for_period(
        data, signal_daily, best, DEVELOPMENT_START, DEVELOPMENT_END
    )
    raw_validation = _entry_signals_for_period(
        data, signal_daily, best, VALIDATION_START, VALIDATION_END
    )
    spread_limit = float(development["SpreadPoints"].quantile(0.90))
    simulator_config = RiskControlConfig("Balanced Robustness", "Entry sensitivity")
    configs = [
        RobustnessConfig(fast, slow, wait, conviction)
        for fast, slow in MA_PAIRS
        for wait in WAIT_HOURS
        for conviction in CONVICTION_MULTIPLIERS
    ]
    feature_cache = {
        (fast, slow): _entry_features(data, h1_fast_span=fast, h1_slow_span=slow)
        for fast, slow in MA_PAIRS
    }

    development_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    center_result = None
    center_signals = pd.DataFrame()
    center_features = feature_cache[(10, 30)]
    for config in configs:
        quality_config = _quality_config(config)
        features = feature_cache[(config.fast_ma, config.slow_ma)]
        selected_development, _ = _select_signals(
            raw_development, features, best, quality_config, spread_limit, DEVELOPMENT_END
        )
        development_result = _simulate_risk_control(
            development, selected_development, best, simulator_config
        )
        development_rows.append(
            _robustness_row(
                config,
                development_result,
                len(raw_development),
                len(selected_development),
                "Development 2025",
            )
        )

        selected_validation, _ = _select_signals(
            raw_validation, features, best, quality_config, spread_limit, VALIDATION_END
        )
        validation_result = _simulate_risk_control(
            validation, selected_validation, best, simulator_config
        )
        validation_rows.append(
            _robustness_row(
                config,
                validation_result,
                len(raw_validation),
                len(selected_validation),
                "Validation 2026H1",
            )
        )
        if _is_center(config):
            center_result = validation_result
            center_signals = selected_validation

    if center_result is None:
        raise RuntimeError("Konfigurasi pusat Balanced Entry tidak ditemukan.")

    development_table = pd.DataFrame(development_rows)
    validation_table = pd.DataFrame(validation_rows)
    center_stress = _stress_center(
        validation,
        center_signals,
        best,
        simulator_config,
    )
    monte_carlo, monte_carlo_summary = _monte_carlo(center_result.trades)
    monthly = _monthly_summary(center_result)
    regime = _regime_summary(center_result.trades, center_features, development)
    direction = _group_trade_summary(center_result.trades, "Arah")
    stability = _stability_summary(validation_table)

    return {
        "methodology": {
            "Baseline lock": "Optimizer v1 live tidak diubah sampai 31 Agustus 2026",
            "Center": "H1 MA 10/30 | wait 2 jam | conviction 1.05",
            "Neighborhood": "3 pasangan MA x 3 wait x 3 conviction = 27 konfigurasi",
            "Development": "01 Jan 2025 - 31 Des 2025",
            "Validation": "01 Jan 2026 - 30 Jun 2026",
            "Purpose": "Uji sensitivitas, bukan optimasi ulang atau pemilihan pemenang baru",
            "Caveat": (
                "2026H1 adalah secondary validation yang telah diamati. Bukti independen tetap berasal "
                "dari forward paper shadow setelah konfigurasi dibekukan."
            ),
        },
        "criteria": {
            "Proporsi growth positif minimum (%)": 70.0,
            "Proporsi profit factor >= 1.30 minimum (%)": 60.0,
            "Proporsi drawdown <= 10% minimum (%)": 60.0,
            "Proporsi tiga kriteria minimum (%)": 50.0,
        },
        "development": development_table,
        "validation": validation_table,
        "stability": stability,
        "center_result": _compact_curve(center_result),
        "center_stress": center_stress,
        "center_monte_carlo": monte_carlo,
        "center_monte_carlo_summary": monte_carlo_summary,
        "center_monthly": monthly,
        "regime_summary": regime,
        "direction_summary": direction,
    }


def _quality_config(config: RobustnessConfig) -> SignalQualityConfig:
    return SignalQualityConfig(
        config.name,
        "Robustness neighborhood",
        conviction_multiplier=config.conviction_multiplier,
        require_h1_trend=True,
        wait_hours=config.wait_hours,
    )


def _is_center(config: RobustnessConfig) -> bool:
    return (
        config.fast_ma,
        config.slow_ma,
        config.wait_hours,
        config.conviction_multiplier,
    ) == CENTER_PARAMETERS


def _robustness_row(
    config: RobustnessConfig,
    result,
    raw_count: int,
    selected_count: int,
    period: str,
) -> dict[str, object]:
    metrics = _metric_values(result)
    retention = selected_count / raw_count * 100 if raw_count else 0.0
    core_pass = bool(
        metrics["Growth (%)"] > 0
        and metrics["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT
        and metrics["Profit factor"] >= PROFIT_FACTOR_TARGET
    )
    return {
        "Konfigurasi": config.name,
        "Periode": period,
        "Fast MA": config.fast_ma,
        "Slow MA": config.slow_ma,
        "Wait (jam)": config.wait_hours,
        "Conviction multiplier": config.conviction_multiplier,
        "Konfigurasi pusat": _is_center(config),
        **metrics,
        "Sinyal awal": raw_count,
        "Entry lolos": selected_count,
        "Retensi entry (%)": retention,
        "Kuartal positif": _positive_quarters(result),
        "Growth positif": metrics["Growth (%)"] > 0,
        "Drawdown <= 10%": metrics["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT,
        "Profit factor >= 1.30": metrics["Profit factor"] >= PROFIT_FACTOR_TARGET,
        "Lolos tiga kriteria": core_pass,
    }


def _stability_summary(validation: pd.DataFrame) -> dict[str, object]:
    total = max(len(validation), 1)

    def percentage(column: str) -> float:
        return float(validation[column].astype(bool).sum() / total * 100)

    values = {
        "Jumlah konfigurasi": int(len(validation)),
        "Growth positif (%)": percentage("Growth positif"),
        "Drawdown <= 10% (%)": percentage("Drawdown <= 10%"),
        "Profit factor >= 1.30 (%)": percentage("Profit factor >= 1.30"),
        "Lolos tiga kriteria (%)": percentage("Lolos tiga kriteria"),
        "Median growth (%)": float(validation["Growth (%)"].median()),
        "Growth P10 (%)": float(validation["Growth (%)"].quantile(0.10)),
        "Median profit factor": float(validation["Profit factor"].median()),
        "Drawdown P90 (%)": float(validation["Max drawdown (%)"].quantile(0.90)),
    }
    values["Robustness status"] = "LULUS" if (
        values["Growth positif (%)"] >= 70
        and values["Drawdown <= 10% (%)"] >= 60
        and values["Profit factor >= 1.30 (%)"] >= 60
        and values["Lolos tiga kriteria (%)"] >= 50
    ) else "BELUM LULUS"
    return values


def _stress_center(
    validation: pd.DataFrame,
    signals: pd.DataFrame,
    best: dict[str, object],
    simulator_config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for spread_multiplier in (1.0, 1.5, 2.0):
        for slippage_points in (2.0, 4.0, 6.0):
            result = _simulate_risk_control(
                validation,
                signals,
                best,
                simulator_config,
                spread_multiplier=spread_multiplier,
                slippage_points=slippage_points,
            )
            rows.append(
                {
                    "Spread multiplier": spread_multiplier,
                    "Slippage points/sisi": slippage_points,
                    **_metric_values(result),
                }
            )
    return pd.DataFrame(rows)


def _regime_summary(
    trades: pd.DataFrame,
    features: pd.DataFrame,
    development: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    development_features = features.loc[development.index.min() : development.index.max()]
    trend_strength = (
        (development_features["h1_fast"] - development_features["h1_slow"]).abs()
        / development_features["atr"]
    ).replace([np.inf, -np.inf], np.nan)
    trend_threshold = float(trend_strength.median())
    volatility_threshold = float(development_features["atr"].median())

    entry_times = pd.DatetimeIndex(pd.to_datetime(trades["Tanggal entry"], errors="coerce"))
    entry_features = features.reindex(entry_times, method="ffill")
    current_strength = (
        (entry_features["h1_fast"] - entry_features["h1_slow"]).abs()
        / entry_features["atr"]
    ).replace([np.inf, -np.inf], np.nan)
    annotated = trades.reset_index(drop=True).copy()
    annotated["Regime tren"] = np.where(
        current_strength.to_numpy() >= trend_threshold, "Trending", "Sideways"
    )
    annotated["Regime volatilitas"] = np.where(
        entry_features["atr"].to_numpy() >= volatility_threshold, "Volatilitas tinggi", "Volatilitas rendah"
    )
    annotated["Regime"] = annotated["Regime tren"] + " | " + annotated["Regime volatilitas"]
    return _group_trade_summary(annotated, "Regime")


def _group_trade_summary(trades: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, subset in trades.groupby(group_column, dropna=False):
        net = pd.to_numeric(subset["Net P/L"], errors="coerce").fillna(0.0)
        rows.append(
            {
                group_column: group,
                "Transaksi": int(len(subset)),
                "Net P/L": float(net.sum()),
                "Win rate (%)": float((net > 0).mean() * 100) if len(net) else np.nan,
                "Profit factor": _profit_factor(net),
                "Rata-rata P/L": float(net.mean()) if len(net) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("Net P/L", ascending=False).reset_index(drop=True)
