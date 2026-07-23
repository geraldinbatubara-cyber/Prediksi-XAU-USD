from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import SLIPPAGE_POINTS, _compact_curve, _prepare_m1
from gold_forecast.strategy_optimizer import _rsi
from gold_forecast.v1_risk_control import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    MAX_DRAWDOWN_PCT,
    MAX_MONTE_CARLO_LOSS_PCT,
    MIN_TRADES,
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
    _completed_bars,
    _entry_features,
    _select_signals,
)


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    adx_max: float
    efficiency_max: float
    choppiness_min: float
    trend_strength_max: float
    slope_max: float
    minimum_sideways_votes: int


@dataclass(frozen=True)
class SidewaysConfig:
    name: str
    edge_fraction: float
    buy_rsi_max: float
    sell_rsi_min: float
    tp_cap_usd: float
    sl_cap_usd: float
    atr_buffer: float = 0.25
    cooldown_hours: int = 4
    time_stop_hours: int = 12


def run_v1_sideways_defense_lab(
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

    features, h1, m15 = _regime_features(data)
    regime_candidates = _regime_candidates(features, development)
    classifier_development = pd.DataFrame(
        [_classifier_metrics(features, config, DEVELOPMENT_START, DEVELOPMENT_END) for config in regime_candidates]
    )
    selected_regime_name = str(
        classifier_development.sort_values(
            ["Macro F1", "Sideways precision", "Balanced accuracy"],
            ascending=False,
        ).iloc[0]["Classifier"]
    )
    selected_regime = next(config for config in regime_candidates if config.name == selected_regime_name)
    classifier_validation = pd.DataFrame(
        [_classifier_metrics(features, selected_regime, VALIDATION_START, VALIDATION_END)]
    )
    states = _regime_states(features, selected_regime)

    spread_limit = float(development["SpreadPoints"].quantile(0.90))
    raw_development = _entry_signals_for_period(
        data, signal_daily, best, DEVELOPMENT_START, DEVELOPMENT_END
    )
    raw_validation = _entry_signals_for_period(
        data, signal_daily, best, VALIDATION_START, VALIDATION_END
    )
    balanced_config = SignalQualityConfig(
        "Balanced Entry Frozen",
        "Trend engine",
        conviction_multiplier=1.05,
        require_h1_trend=True,
        wait_hours=2,
    )
    balanced_development, _ = _select_signals(
        raw_development, features, best, balanced_config, spread_limit, DEVELOPMENT_END
    )
    balanced_validation, _ = _select_signals(
        raw_validation, features, best, balanced_config, spread_limit, VALIDATION_END
    )
    balanced_development = _label_signals(balanced_development, "Balanced Original")
    balanced_validation = _label_signals(balanced_validation, "Balanced Original")
    gated_development = _gate_trend_signals(balanced_development, states)
    gated_validation = _gate_trend_signals(balanced_validation, states)

    sideways_candidates = _sideways_candidates()
    simulator_config = RiskControlConfig(
        "Regime Strategy",
        "Sideways defense",
        max_total_positions=1,
        max_same_direction=1,
    )
    strategy_development_rows: list[dict[str, object]] = []
    development_results = {}
    for sideways_config in sideways_candidates:
        side_signals = _sideways_signals(
            data,
            features,
            m15,
            states,
            best,
            sideways_config,
            DEVELOPMENT_START,
            DEVELOPMENT_END,
        )
        hybrid_signals = _merge_signals(gated_development, side_signals)
        for strategy_name, signals in (
            (f"Sideways Only | {sideways_config.name}", side_signals),
            (f"Hybrid | {sideways_config.name}", hybrid_signals),
        ):
            result = _simulate_risk_control(
                development, signals, best, simulator_config
            )
            development_results[strategy_name] = result
            strategy_development_rows.append(
                _strategy_row(strategy_name, result, sideways_config.name)
            )
    strategy_development = pd.DataFrame(strategy_development_rows)
    hybrid_development = strategy_development[
        strategy_development["Strategi"].str.startswith("Hybrid |")
    ].copy()
    hybrid_development["Development score"] = hybrid_development.apply(
        _development_score, axis=1
    )
    eligible_hybrid = hybrid_development[
        (hybrid_development["Growth (%)"] > 0)
        & (hybrid_development["Profit factor"] >= 1.20)
        & (hybrid_development["Max drawdown (%)"] <= 15)
    ]
    development_selection_fallback = eligible_hybrid.empty
    selection_pool = eligible_hybrid if not eligible_hybrid.empty else hybrid_development
    selected_sideways_name = str(
        selection_pool.sort_values(
            ["Growth (%)", "Profit factor", "Max drawdown (%)"],
            ascending=[False, False, True],
        ).iloc[0]["Sideways config"]
    )
    selected_sideways = next(
        config for config in sideways_candidates if config.name == selected_sideways_name
    )

    side_validation = _sideways_signals(
        data,
        features,
        m15,
        states,
        best,
        selected_sideways,
        VALIDATION_START,
        VALIDATION_END,
    )
    hybrid_validation = _merge_signals(gated_validation, side_validation)
    validation_signals = {
        "Balanced Entry Frozen": balanced_validation,
        "Trend + Skip Sideways": gated_validation,
        "Sideways Mean Reversion": side_validation,
        "Hybrid Regime Strategy": hybrid_validation,
    }
    validation_results = {
        name: _simulate_risk_control(validation, signals, best, simulator_config)
        for name, signals in validation_signals.items()
    }
    strategy_validation = pd.DataFrame(
        [_strategy_row(name, result, selected_sideways_name) for name, result in validation_results.items()]
    )
    hybrid_result = validation_results["Hybrid Regime Strategy"]
    balanced_result = validation_results["Balanced Entry Frozen"]
    attribution = _strategy_attribution(hybrid_result.trades)
    selected_sideways_development = strategy_development[
        strategy_development["Strategi"].eq(f"Sideways Only | {selected_sideways_name}")
    ].iloc[0]
    selected_classifier_development = classifier_development[
        classifier_development["Classifier"].eq(selected_regime_name)
    ].iloc[0]
    decision = _decision_summary(
        hybrid_result,
        balanced_result,
        attribution,
        validation_results["Sideways Mean Reversion"],
        selected_sideways_development,
        selected_classifier_development,
        development_selection_fallback,
    )
    stress = _stress_hybrid(
        validation,
        hybrid_validation,
        best,
        simulator_config,
    )
    monte_carlo, monte_carlo_summary = _monte_carlo(hybrid_result.trades)
    decision["Stress profitable 9/9"] = bool(
        len(stress) == 9 and (stress["Growth (%)"] > 0).all()
    )
    decision["Monte Carlo rugi <= 10%"] = bool(
        monte_carlo_summary["Probabilitas equity akhir < modal awal (%)"]
        <= MAX_MONTE_CARLO_LOSS_PCT
    )
    decision["Lulus seluruh kriteria"] = all(
        value for key, value in decision.items() if key.startswith("Lolos:")
    ) and decision["Stress profitable 9/9"] and decision["Monte Carlo rugi <= 10%"]

    return {
        "methodology": {
            "Baseline lock": "Optimizer v1 live tidak diubah sampai 31 Agustus 2026",
            "Development": "01 Jan 2025 - 31 Des 2025",
            "Validation": "01 Jan 2026 - 30 Jun 2026",
            "Regime states": "TRENDING | SIDEWAYS | UNCERTAIN",
            "Trend engine": "Balanced Entry H1 10/30, wait 2 jam, conviction 1.05",
            "Sideways engine": "Range H1 + reversal M15 + RSI + ATR invalidation + time stop",
            "Selected classifier": selected_regime_name,
            "Selected sideways config": selected_sideways_name,
            "Development selection fallback": development_selection_fallback,
            "Caveat": (
                "Classifier dan konfigurasi sideways dipilih hanya pada 2025. 2026H1 tetap secondary "
                "validation karena sudah pernah diamati; forward paper shadow adalah bukti independen."
            ),
        },
        "classifier_development": classifier_development,
        "classifier_validation": classifier_validation,
        "state_distribution": _state_distribution(states),
        "strategy_development": strategy_development,
        "strategy_validation": strategy_validation,
        "decision": decision,
        "selected_regime_config": asdict(selected_regime),
        "selected_sideways_config": asdict(selected_sideways),
        "strategy_attribution": attribution,
        "hybrid_result": _compact_curve(hybrid_result),
        "balanced_result": _compact_curve(balanced_result),
        "hybrid_monthly": _monthly_summary(hybrid_result),
        "hybrid_stress": stress,
        "hybrid_monte_carlo": monte_carlo,
        "hybrid_monte_carlo_summary": monte_carlo_summary,
        "signal_counts": {
            name: int(len(signals)) for name, signals in validation_signals.items()
        },
    }


def _regime_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _entry_features(data, h1_fast_span=10, h1_slow_span=30)
    h1 = _completed_bars(data, "1h")
    high, low, close = h1["High"], h1["Low"], h1["Close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=h1.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=h1.index
    )
    plus_di = 100 * plus_dm.rolling(14).sum() / true_range.rolling(14).sum()
    minus_di = 100 * minus_dm.rolling(14).sum() / true_range.rolling(14).sum()
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    h1["adx"] = dx.rolling(14, min_periods=8).mean()
    travel = close.diff().abs().rolling(14).sum()
    h1["efficiency"] = close.diff(14).abs() / travel
    range_sum = true_range.rolling(14).sum()
    range_span = high.rolling(14).max() - low.rolling(14).min()
    h1["choppiness"] = 100 * np.log10(range_sum / range_span) / np.log10(14)
    h1["atr"] = atr
    h1["h1_fast"] = close.ewm(span=10, adjust=False).mean()
    h1["h1_slow"] = close.ewm(span=30, adjust=False).mean()
    h1["trend_strength"] = (h1["h1_fast"] - h1["h1_slow"]).abs() / atr
    h1["slope"] = h1["h1_slow"].diff(3).abs() / atr
    h1["range_high"] = high.shift(1).rolling(20).max()
    h1["range_low"] = low.shift(1).rolling(20).min()
    h1["range_mid"] = (h1["range_high"] + h1["range_low"]) / 2
    h1["future_efficiency"], h1["future_move_atr"] = _future_path_labels(close, atr, 6)

    m15 = _completed_bars(data, "15min")
    m15["rsi"] = _rsi(m15["Close"], 14)
    m15["bullish_reversal"] = m15["Close"] > m15["Open"]
    m15["bearish_reversal"] = m15["Close"] < m15["Open"]

    h1_columns = [
        "adx", "efficiency", "choppiness", "atr", "h1_fast", "h1_slow",
        "trend_strength", "slope", "range_high", "range_low", "range_mid",
        "future_efficiency", "future_move_atr",
    ]
    for column in h1_columns:
        base[column] = h1[column].reindex(base.index, method="ffill")
    return base.replace([np.inf, -np.inf], np.nan), h1, m15


def _future_path_labels(
    close: pd.Series,
    atr: pd.Series,
    horizon: int,
) -> tuple[pd.Series, pd.Series]:
    steps = [(close.shift(-offset) - close.shift(-(offset - 1))).abs() for offset in range(1, horizon + 1)]
    travel = pd.concat(steps, axis=1).sum(axis=1, min_count=horizon)
    displacement = (close.shift(-horizon) - close).abs()
    return displacement / travel, displacement / atr


def _regime_candidates(features: pd.DataFrame, development: pd.DataFrame) -> list[RegimeConfig]:
    dev = features.loc[development.index.min() : development.index.max()]
    return [
        RegimeConfig("RG-A Conservative", 18.0, 0.25, 60.0, 0.25, 0.08, 4),
        RegimeConfig("RG-B Balanced", 20.0, 0.30, 58.0, 0.35, 0.12, 3),
        RegimeConfig("RG-C Sensitive", 22.0, 0.35, 55.0, 0.45, 0.16, 3),
        RegimeConfig(
            "RG-D Development Quantile",
            float(dev["adx"].quantile(0.35)),
            float(dev["efficiency"].quantile(0.35)),
            float(dev["choppiness"].quantile(0.65)),
            float(dev["trend_strength"].quantile(0.35)),
            float(dev["slope"].quantile(0.35)),
            3,
        ),
    ]


def _regime_states(features: pd.DataFrame, config: RegimeConfig) -> pd.Series:
    checks = pd.DataFrame(index=features.index)
    checks["adx"] = features["adx"] <= config.adx_max
    checks["efficiency"] = features["efficiency"] <= config.efficiency_max
    checks["choppiness"] = features["choppiness"] >= config.choppiness_min
    checks["trend_strength"] = features["trend_strength"] <= config.trend_strength_max
    checks["slope"] = features["slope"] <= config.slope_max
    votes = checks.fillna(False).sum(axis=1)
    raw_sideways = votes >= config.minimum_sideways_votes
    raw_trending = votes <= 1
    sideways_confirmed = raw_sideways.rolling(120, min_periods=120).sum() == 120
    trending_confirmed = raw_trending.rolling(120, min_periods=120).sum() == 120
    states = pd.Series("UNCERTAIN", index=features.index, dtype="object")
    states.loc[sideways_confirmed] = "SIDEWAYS"
    states.loc[trending_confirmed] = "TRENDING"
    return states


def _truth_labels(features: pd.DataFrame) -> pd.Series:
    truth = pd.Series("UNCERTAIN", index=features.index, dtype="object")
    truth.loc[
        (features["future_efficiency"] <= 0.35)
        & (features["future_move_atr"] <= 0.80)
    ] = "SIDEWAYS"
    truth.loc[
        (features["future_efficiency"] >= 0.55)
        & (features["future_move_atr"] >= 0.80)
    ] = "TRENDING"
    return truth


def _classifier_metrics(
    features: pd.DataFrame,
    config: RegimeConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    states = _regime_states(features, config).loc[start:end]
    truth = _truth_labels(features).loc[start:end]
    hourly = pd.DataFrame({"Prediksi": states, "Aktual": truth}).resample("1h").last().dropna()
    evaluated = hourly[hourly["Aktual"].isin(["SIDEWAYS", "TRENDING"])]
    side_actual = evaluated["Aktual"].eq("SIDEWAYS")
    side_predicted = evaluated["Prediksi"].eq("SIDEWAYS")
    trend_actual = evaluated["Aktual"].eq("TRENDING")
    trend_predicted = evaluated["Prediksi"].eq("TRENDING")
    side_precision = _safe_ratio((side_actual & side_predicted).sum(), side_predicted.sum())
    side_recall = _safe_ratio((side_actual & side_predicted).sum(), side_actual.sum())
    trend_precision = _safe_ratio((trend_actual & trend_predicted).sum(), trend_predicted.sum())
    trend_recall = _safe_ratio((trend_actual & trend_predicted).sum(), trend_actual.sum())
    side_f1 = _f1(side_precision, side_recall)
    trend_f1 = _f1(trend_precision, trend_recall)
    return {
        "Classifier": config.name,
        "Observasi berlabel": int(len(evaluated)),
        "Coverage keputusan (%)": float(evaluated["Prediksi"].ne("UNCERTAIN").mean() * 100),
        "Sideways precision": side_precision,
        "Sideways recall": side_recall,
        "Trend precision": trend_precision,
        "Trend recall": trend_recall,
        "Balanced accuracy": (side_recall + trend_recall) / 2,
        "Macro F1": (side_f1 + trend_f1) / 2,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _sideways_candidates() -> list[SidewaysConfig]:
    return [
        SidewaysConfig("MR-A Conservative", 0.12, 35.0, 65.0, 12.0, 10.0),
        SidewaysConfig("MR-B Balanced", 0.20, 40.0, 60.0, 15.0, 10.0),
        SidewaysConfig("MR-C Wide", 0.25, 45.0, 55.0, 15.0, 12.0),
    ]


def _sideways_signals(
    data: pd.DataFrame,
    features: pd.DataFrame,
    m15: pd.DataFrame,
    states: pd.Series,
    best: dict[str, object],
    config: SidewaysConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    last_signal = pd.Timestamp.min
    threshold = float(best["Threshold entry (%)"])
    for timestamp, candle in m15.loc[start:end].iterrows():
        if timestamp not in features.index or states.get(timestamp, "UNCERTAIN") != "SIDEWAYS":
            continue
        if timestamp < last_signal + pd.Timedelta(hours=config.cooldown_hours):
            continue
        row = features.loc[timestamp]
        range_high = float(row.get("range_high", np.nan))
        range_low = float(row.get("range_low", np.nan))
        range_mid = float(row.get("range_mid", np.nan))
        atr = float(row.get("atr", np.nan))
        close = float(candle["Close"])
        rsi = float(candle.get("rsi", np.nan))
        if not np.isfinite([range_high, range_low, range_mid, atr, close, rsi]).all():
            continue
        width = range_high - range_low
        if width <= 0 or width > 6 * atr:
            continue
        direction = None
        if (
            close <= range_low + config.edge_fraction * width
            and rsi <= config.buy_rsi_max
            and bool(candle["bullish_reversal"])
        ):
            direction = "BUY"
            tp_usd = min(max(range_mid - close, 5.0), config.tp_cap_usd)
            sl_usd = min(max(close - (range_low - config.atr_buffer * atr), 5.0), config.sl_cap_usd)
        elif (
            close >= range_high - config.edge_fraction * width
            and rsi >= config.sell_rsi_min
            and bool(candle["bearish_reversal"])
        ):
            direction = "SELL"
            tp_usd = min(max(close - range_mid, 5.0), config.tp_cap_usd)
            sl_usd = min(max((range_high + config.atr_buffer * atr) - close, 5.0), config.sl_cap_usd)
        if direction is None or tp_usd / sl_usd < 0.80:
            continue
        location = data.index.searchsorted(timestamp, side="right")
        if location >= len(data.index):
            continue
        entry_time = pd.Timestamp(data.index[location])
        if entry_time > end or states.get(entry_time, "UNCERTAIN") != "SIDEWAYS":
            continue
        sign = 1 if direction == "BUY" else -1
        expected = sign * (threshold + 0.01)
        reference = float(data.loc[entry_time, "Close"])
        rows.append(
            {
                "entry_time": entry_time,
                "signal_date": timestamp.normalize(),
                "prediction": reference * (1 + expected / 100),
                "expected_change_pct": expected,
                "lot": 0.01,
                "tp_usd": tp_usd,
                "sl_usd": sl_usd,
                "time_stop_hours": config.time_stop_hours,
                "strategy": "Sideways Mean Reversion",
            }
        )
        last_signal = timestamp
    if not rows:
        return pd.DataFrame(
            columns=["signal_date", "prediction", "expected_change_pct", "lot", "tp_usd", "sl_usd", "time_stop_hours", "strategy"]
        )
    return pd.DataFrame(rows).set_index("entry_time").sort_index()


def _label_signals(signals: pd.DataFrame, strategy: str) -> pd.DataFrame:
    labeled = signals.copy()
    labeled["strategy"] = strategy
    return labeled


def _gate_trend_signals(signals: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    keep = [states.get(pd.Timestamp(timestamp), "UNCERTAIN") == "TRENDING" for timestamp in signals.index]
    gated = signals.loc[keep].copy()
    gated["strategy"] = "Balanced Trend"
    return gated


def _merge_signals(trend: pd.DataFrame, sideways: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([trend, sideways], axis=0, sort=False).sort_index()
    return merged.loc[~merged.index.duplicated(keep="first")]


def _strategy_row(strategy: str, result, sideways_config: str) -> dict[str, object]:
    return {
        "Strategi": strategy,
        "Sideways config": sideways_config,
        **_metric_values(result),
    }


def _development_score(row: pd.Series) -> float:
    growth = max(float(row["Growth (%)"]), 0.0)
    drawdown = float(row["Max drawdown (%)"])
    profit_factor = float(row["Profit factor"])
    transactions = float(row["Transaksi"])
    return (
        30 * min(growth / 20, 1)
        + 30 * min(max(profit_factor, 0) / PROFIT_FACTOR_TARGET, 1)
        + 25 * max(0, 1 - drawdown / 15)
        + 15 * min(transactions / 50, 1)
    )


def _strategy_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades.empty or "Strategi" not in trades.columns:
        return pd.DataFrame()
    for strategy, subset in trades.groupby("Strategi"):
        net = pd.to_numeric(subset["Net P/L"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "Strategi": strategy,
                "Transaksi": int(len(subset)),
                "Net P/L": float(net.sum()),
                "Win rate (%)": float((net > 0).mean() * 100),
                "Profit factor": _profit_factor(net),
            }
        )
    return pd.DataFrame(rows).sort_values("Net P/L", ascending=False).reset_index(drop=True)


def _decision_summary(
    hybrid_result,
    balanced_result,
    attribution: pd.DataFrame,
    sideways_result,
    sideways_development: pd.Series,
    classifier_development: pd.Series,
    development_selection_fallback: bool,
) -> dict[str, object]:
    hybrid = _metric_values(hybrid_result)
    balanced = _metric_values(balanced_result)
    sideways = _metric_values(sideways_result)
    trend_row = attribution[attribution["Strategi"].eq("Balanced Trend")]
    trend_net = float(trend_row.iloc[0]["Net P/L"]) if not trend_row.empty else 0.0
    balanced_net = float(balanced_result.summary["Total net P/L"])
    trend_retention = trend_net / balanced_net * 100 if balanced_net > 0 else 0.0
    return {
        "Lolos: Classifier Macro F1 >= 0.50": bool(classifier_development["Macro F1"] >= 0.50),
        "Lolos: Sideways development positif": bool(sideways_development["Growth (%)"] > 0),
        "Lolos: Sideways development PF >= 1.20": bool(sideways_development["Profit factor"] >= 1.20),
        "Lolos: Tidak memakai fallback development": not development_selection_fallback,
        "Lolos: Growth positif": bool(hybrid["Growth (%)"] > 0),
        "Lolos: Drawdown <= 10%": bool(hybrid["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT),
        "Lolos: Profit factor >= 1.30": bool(hybrid["Profit factor"] >= PROFIT_FACTOR_TARGET),
        "Lolos: Sideways net positif": bool(sideways["Growth (%)"] > 0),
        "Lolos: Sideways PF >= 1.30": bool(sideways["Profit factor"] >= PROFIT_FACTOR_TARGET),
        "Lolos: Profit trend dipertahankan >= 85%": bool(trend_retention >= 85),
        "Lolos: Transaksi >= 50": bool(hybrid["Transaksi"] >= MIN_TRADES),
        "Hybrid growth (%)": hybrid["Growth (%)"],
        "Hybrid drawdown (%)": hybrid["Max drawdown (%)"],
        "Hybrid profit factor": hybrid["Profit factor"],
        "Hybrid transaksi": hybrid["Transaksi"],
        "Sideways growth (%)": sideways["Growth (%)"],
        "Sideways profit factor": sideways["Profit factor"],
        "Sideways development growth (%)": float(sideways_development["Growth (%)"]),
        "Sideways development profit factor": float(sideways_development["Profit factor"]),
        "Classifier development Macro F1": float(classifier_development["Macro F1"]),
        "Profit trend dipertahankan (%)": trend_retention,
    }


def _stress_hybrid(
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


def _state_distribution(states: pd.Series) -> pd.DataFrame:
    rows = []
    for period, start, end in (
        ("Development 2025", DEVELOPMENT_START, DEVELOPMENT_END),
        ("Validation 2026H1", VALIDATION_START, VALIDATION_END),
    ):
        hourly = states.loc[start:end].resample("1h").last().dropna()
        counts = hourly.value_counts()
        for state in ("TRENDING", "SIDEWAYS", "UNCERTAIN"):
            count = int(counts.get(state, 0))
            rows.append(
                {
                    "Periode": period,
                    "Regime": state,
                    "Jam": count,
                    "Proporsi (%)": count / max(len(hourly), 1) * 100,
                }
            )
    return pd.DataFrame(rows)
