from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import SLIPPAGE_POINTS, _compact_curve, _prepare_m1
from gold_forecast.strategy_optimizer import MultiPhaseSimulationResult
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
from gold_forecast.v1_robustness import _monte_carlo, _monthly_summary


@dataclass(frozen=True)
class SignalQualityConfig:
    name: str
    group: str
    conviction_multiplier: float = 1.0
    require_m15_trend: bool = False
    require_h1_trend: bool = False
    require_h1_momentum: bool = False
    max_stretch_atr: float | None = None
    spread_quantile: float | None = None
    wait_hours: int = 0
    minimum_confirmations: int = 0
    require_h1_primary: bool = False


def run_v1_signal_quality_lab(
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

    features = _entry_features(data)
    raw_development = _entry_signals_for_period(
        data, signal_daily, best, DEVELOPMENT_START, DEVELOPMENT_END
    )
    raw_validation = _entry_signals_for_period(
        data, signal_daily, best, VALIDATION_START, VALIDATION_END
    )
    spread_limit = float(development["SpreadPoints"].quantile(0.90))
    candidates = _candidate_configs()
    baseline = SignalQualityConfig("v1 Exact Baseline", "Baseline")
    simulator_config = RiskControlConfig("Signal Quality", "Entry filter")

    development_results: dict[str, MultiPhaseSimulationResult] = {}
    development_rows: list[dict[str, object]] = []
    audit_frames: dict[str, pd.DataFrame] = {}
    for config in [baseline, *candidates]:
        selected, audit = _select_signals(
            raw_development,
            features,
            best,
            config,
            spread_limit,
            DEVELOPMENT_END,
        )
        result = _simulate_risk_control(development, selected, best, simulator_config)
        development_results[config.name] = result
        audit_frames[config.name] = audit
        development_rows.append(
            _summary_row(config, result, len(raw_development), len(selected), "Development 2025")
        )

    development_table = pd.DataFrame(development_rows)
    baseline_development = development_results[baseline.name]
    development_table["Pre-score"] = development_table.apply(
        lambda row: _development_score(row, baseline_development.summary), axis=1
    )
    balanced = development_table[development_table["Kelompok"].eq("Balanced v3")]
    eligible = balanced[balanced["Development eligible"]]
    selection_fallback = eligible.empty
    finalist_pool = eligible if not eligible.empty else balanced
    finalist_names = (
        finalist_pool
        .sort_values(
            ["Pre-score", "Kuartal positif", "Profit factor", "Max drawdown (%)"],
            ascending=[False, False, False, True],
        )
        .head(5)["Kandidat"]
        .tolist()
    )
    config_by_name = {config.name: config for config in candidates}

    baseline_validation_signals, baseline_audit = _select_signals(
        raw_validation, features, best, baseline, spread_limit, VALIDATION_END
    )
    baseline_validation = _simulate_risk_control(
        validation, baseline_validation_signals, best, simulator_config
    )
    validation_results = {baseline.name: baseline_validation}
    validation_rows = [
        _summary_row(
            baseline,
            baseline_validation,
            len(raw_validation),
            len(baseline_validation_signals),
            "Validation 2026H1",
        )
    ]
    validation_audits = {baseline.name: baseline_audit}
    stress_rows: list[dict[str, object]] = []
    monte_carlo_rows: list[dict[str, object]] = []
    monte_carlo_frames: dict[str, pd.DataFrame] = {}
    monthly_frames: dict[str, pd.DataFrame] = {}

    reference_names = [
        config.name for config in candidates if config.group == "Reference v1"
    ]
    for name in [*reference_names, *finalist_names]:
        config = config_by_name[name]
        selected, audit = _select_signals(
            raw_validation, features, best, config, spread_limit, VALIDATION_END
        )
        result = _simulate_risk_control(validation, selected, best, simulator_config)
        validation_results[name] = result
        validation_audits[name] = audit
        validation_rows.append(
            _summary_row(config, result, len(raw_validation), len(selected), "Validation 2026H1")
        )
        if name in reference_names:
            continue
        monthly_frames[name] = _monthly_summary(result)
        mc_frame, mc_summary = _monte_carlo(result.trades)
        monte_carlo_frames[name] = mc_frame
        monte_carlo_rows.append({"Kandidat": name, **mc_summary})
        for spread_multiplier in (1.0, 1.5, 2.0):
            for slippage_points in (2.0, 4.0, 6.0):
                stressed = _simulate_risk_control(
                    validation,
                    selected,
                    best,
                    simulator_config,
                    spread_multiplier=spread_multiplier,
                    slippage_points=slippage_points,
                )
                stress_rows.append(
                    {
                        "Kandidat": name,
                        "Spread multiplier": spread_multiplier,
                        "Slippage points/sisi": slippage_points,
                        **_metric_values(stressed),
                    }
                )

    validation_table = pd.DataFrame(validation_rows)
    stress_table = pd.DataFrame(stress_rows)
    monte_carlo_table = pd.DataFrame(monte_carlo_rows)
    decision = _decision_table(
        validation_table, stress_table, monte_carlo_table, monthly_frames
    )
    ranked = decision.sort_values(
        ["Lulus seluruh kriteria", "Jumlah kriteria lolos", "Final score"],
        ascending=[False, False, False],
    )
    winner_name = str(ranked.iloc[0]["Kandidat"])
    winner_result = validation_results[winner_name]
    winner_config = config_by_name[winner_name]
    winner_passed = bool(
        ranked.iloc[0]["Lulus seluruh kriteria"] and not selection_fallback
    )

    return {
        "methodology": {
            "Baseline lock": "Optimizer v1 live tidak diubah sampai 31 Agustus 2026",
            "Development": "01 Jan 2025 - 31 Des 2025",
            "Validation": "01 Jan 2026 - 30 Jun 2026",
            "Core strategy": str(best["Strategi"]),
            "Yang diubah": "Filter dan timing entry saja; exit, TP/SL, lot, swap, dan fase tetap v1",
            "Spread P90 development": spread_limit,
            "Finalists": finalist_names,
            "Reference": reference_names,
            "Selection rule": (
                "Retensi development 40-75%, minimal 3/4 kuartal positif, growth positif, "
                "profit factor >= 1.25, dan drawdown <= 15%"
            ),
            "Validation status": (
                "Secondary validation; 2026H1 sudah pernah diamati pada eksperimen sebelumnya. "
                "Pemenang wajib menjalani forward paper shadow dan tidak menggantikan baseline terkunci."
            ),
            "Selection fallback": selection_fallback,
        },
        "criteria": {
            "Growth minimum (%)": 0.0,
            "Max drawdown maksimum (%)": MAX_DRAWDOWN_PCT,
            "Profit factor minimum": PROFIT_FACTOR_TARGET,
            "Monte Carlo rugi maksimum (%)": MAX_MONTE_CARLO_LOSS_PCT,
            "Minimum transaksi": MIN_TRADES,
            "Stress profitable": "9/9",
        },
        "development": development_table,
        "validation": validation_table,
        "stress": stress_table,
        "monte_carlo_summary": monte_carlo_table,
        "decision": decision,
        "winner_name": winner_name,
        "winner_status": "LULUS" if winner_passed else "BELUM LULUS",
        "winner_result": _compact_curve(winner_result),
        "winner_monthly": monthly_frames[winner_name],
        "winner_monte_carlo": monte_carlo_frames[winner_name],
        "winner_config": asdict(winner_config),
        "winner_entry_audit": validation_audits[winner_name],
        "baseline_validation": _metric_values(baseline_validation),
        "baseline_match_note": (
            "Baris baseline memakai simulator Exact v1 yang sama. Kandidat hanya boleh mengubah daftar "
            "dan waktu entry, bukan rule keluar atau pengelolaan posisi v1."
        ),
    }


def _candidate_configs() -> list[SignalQualityConfig]:
    return [
        SignalQualityConfig(
            "SQ-A v1 Strict Reference",
            "Reference v1",
            conviction_multiplier=1.15,
            require_m15_trend=True,
            require_h1_trend=True,
        ),
        SignalQualityConfig(
            "BE-A 2of3 Immediate",
            "Balanced v3",
            conviction_multiplier=1.0,
            minimum_confirmations=2,
        ),
        SignalQualityConfig(
            "BE-B M15 Trend Wait-2h",
            "Balanced v3",
            conviction_multiplier=1.05,
            require_m15_trend=True,
            wait_hours=2,
        ),
        SignalQualityConfig(
            "BE-C 2of3 ATR Guard",
            "Balanced v3",
            conviction_multiplier=1.0,
            minimum_confirmations=2,
            max_stretch_atr=2.0,
        ),
        SignalQualityConfig(
            "BE-D H1 Anchor Wait-1h",
            "Balanced v3",
            conviction_multiplier=1.0,
            minimum_confirmations=2,
            require_h1_primary=True,
            wait_hours=1,
        ),
        SignalQualityConfig(
            "BE-E H1 Trend Wait-2h",
            "Balanced v3",
            conviction_multiplier=1.05,
            require_h1_trend=True,
            wait_hours=2,
        ),
        SignalQualityConfig(
            "BE-F 2of3 Wait-1h",
            "Balanced v3",
            conviction_multiplier=1.05,
            minimum_confirmations=2,
            wait_hours=1,
        ),
        SignalQualityConfig(
            "BE-G H1 Trend Immediate",
            "Balanced v3",
            conviction_multiplier=1.05,
            require_h1_trend=True,
        ),
        SignalQualityConfig(
            "BE-H H1 Spread Guard",
            "Balanced v3",
            conviction_multiplier=1.05,
            require_h1_trend=True,
            spread_quantile=0.90,
        ),
    ]


def _entry_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    features["price"] = data["Close"].shift(1)
    features["spread_points"] = data["SpreadPoints"].shift(1)

    m15 = _completed_bars(data, "15min")
    m15_features = pd.DataFrame(index=m15.index)
    m15_features["m15_fast"] = m15["Close"].ewm(span=12, adjust=False).mean()
    m15_features["m15_slow"] = m15["Close"].ewm(span=48, adjust=False).mean()
    m15_features["m15_momentum"] = m15["Close"].pct_change(4) * 100

    h1 = _completed_bars(data, "1h")
    h1_features = pd.DataFrame(index=h1.index)
    h1_features["h1_fast"] = h1["Close"].ewm(span=10, adjust=False).mean()
    h1_features["h1_slow"] = h1["Close"].ewm(span=30, adjust=False).mean()
    h1_features["h1_momentum"] = h1["Close"].pct_change(6) * 100
    h1_true_range = pd.concat(
        [
            h1["High"] - h1["Low"],
            (h1["High"] - h1["Close"].shift(1)).abs(),
            (h1["Low"] - h1["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    h1_features["atr"] = h1_true_range.rolling(14, min_periods=6).mean()

    features = features.join(m15_features.reindex(features.index, method="ffill"))
    features = features.join(h1_features.reindex(features.index, method="ffill"))
    features["stretch_atr"] = (features["price"] - features["h1_fast"]).abs() / features["atr"]
    return features.replace([np.inf, -np.inf], np.nan)


def _completed_bars(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        data[["Open", "High", "Low", "Close"]]
        .resample(rule, label="right", closed="left")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )


def _select_signals(
    signals: pd.DataFrame,
    features: pd.DataFrame,
    best: dict[str, object],
    config: SignalQualityConfig,
    spread_limit: float,
    period_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected: list[pd.Series] = []
    audit: list[dict[str, object]] = []
    threshold = float(best["Threshold entry (%)"]) * config.conviction_multiplier
    for entry_time, signal in signals.iterrows():
        expected = float(signal["expected_change_pct"])
        direction = "BUY" if expected > 0 else "SELL"
        conviction_ok = abs(expected) >= threshold
        deadline = min(pd.Timestamp(entry_time) + pd.Timedelta(hours=config.wait_hours), period_end)
        candidate_times = features.loc[pd.Timestamp(entry_time) : deadline].index
        if config.wait_hours == 0:
            candidate_times = candidate_times[:1]
        chosen_time: pd.Timestamp | None = None
        failure = "Conviction di bawah ambang"
        if conviction_ok:
            for timestamp in candidate_times:
                row = features.loc[timestamp]
                checks = _quality_checks(row, direction, config, spread_limit)
                if all(checks.values()):
                    chosen_time = pd.Timestamp(timestamp)
                    failure = "Lolos"
                    break
                failure = ", ".join(name for name, passed in checks.items() if not passed)
        if chosen_time is not None:
            accepted = signal.copy()
            accepted.name = chosen_time
            selected.append(accepted)
        audit.append(
            {
                "Waktu sinyal awal": pd.Timestamp(entry_time),
                "Waktu entry terpilih": chosen_time,
                "Arah": direction,
                "Expected change (%)": expected,
                "Status": "LOLOS" if chosen_time is not None else "DITOLAK",
                "Alasan": failure,
            }
        )
    if selected:
        selected_frame = pd.DataFrame(selected)
        selected_frame.index.name = "entry_time"
        selected_frame = selected_frame.sort_index()
    else:
        selected_frame = signals.iloc[0:0].copy()
    return selected_frame, pd.DataFrame(audit)


def _quality_checks(
    row: pd.Series,
    direction: str,
    config: SignalQualityConfig,
    spread_limit: float,
) -> dict[str, bool]:
    sign = 1 if direction == "BUY" else -1
    price = float(row.get("price", np.nan))
    m15_fast = float(row.get("m15_fast", np.nan))
    m15_slow = float(row.get("m15_slow", np.nan))
    h1_fast = float(row.get("h1_fast", np.nan))
    h1_slow = float(row.get("h1_slow", np.nan))
    m15_aligned = bool(
        np.isfinite([price, m15_fast, m15_slow]).all()
        and sign * (price - m15_fast) > 0
        and sign * (m15_fast - m15_slow) > 0
        and sign * float(row.get("m15_momentum", np.nan)) > 0
    )
    h1_aligned = bool(
        np.isfinite([price, h1_fast, h1_slow]).all()
        and sign * (price - h1_fast) > 0
        and sign * (h1_fast - h1_slow) > 0
    )
    h1_momentum_aligned = bool(sign * float(row.get("h1_momentum", np.nan)) > 0)
    checks: dict[str, bool] = {}
    if config.require_m15_trend:
        checks["M15 tidak searah"] = m15_aligned
    if config.require_h1_trend:
        checks["H1 tidak searah"] = h1_aligned
    if config.require_h1_momentum:
        checks["Momentum H1 tidak searah"] = h1_momentum_aligned
    if config.require_h1_primary:
        checks["Jangkar tren H1 belum searah"] = h1_aligned
    if config.minimum_confirmations:
        confirmation_count = sum((m15_aligned, h1_aligned, h1_momentum_aligned))
        checks[f"Konfirmasi kurang dari {config.minimum_confirmations}/3"] = bool(
            confirmation_count >= config.minimum_confirmations
        )
    if config.max_stretch_atr is not None:
        checks["Harga terlalu jauh dari tren"] = bool(
            float(row.get("stretch_atr", np.inf)) <= config.max_stretch_atr
        )
    if config.spread_quantile is not None:
        checks["Spread terlalu lebar"] = bool(
            float(row.get("spread_points", np.inf)) <= spread_limit
        )
    return checks


def _summary_row(
    config: SignalQualityConfig,
    result: MultiPhaseSimulationResult,
    raw_count: int,
    selected_count: int,
    period: str,
) -> dict[str, object]:
    metrics = _metric_values(result)
    positive_quarters = _positive_quarters(result)
    retention = selected_count / raw_count * 100 if raw_count else 0.0
    initial_pass = bool(
        metrics["Growth (%)"] > 0
        and metrics["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT
        and metrics["Profit factor"] >= PROFIT_FACTOR_TARGET
        and metrics["Transaksi"] >= MIN_TRADES
    )
    development_eligible = bool(
        period == "Development 2025"
        and 40 <= retention <= 75
        and positive_quarters >= 3
        and metrics["Growth (%)"] > 0
        and metrics["Profit factor"] >= 1.25
        and metrics["Max drawdown (%)"] <= 15
    )
    return {
        "Kandidat": config.name,
        "Kelompok": config.group,
        "Periode": period,
        **metrics,
        "Sinyal awal": raw_count,
        "Entry lolos": selected_count,
        "Entry ditolak": raw_count - selected_count,
        "Retensi entry (%)": retention,
        "Kuartal positif": positive_quarters,
        "Development eligible": development_eligible,
        "Kriteria awal lolos": initial_pass,
        "Konfigurasi": _config_label(config),
    }


def _positive_quarters(result: MultiPhaseSimulationResult) -> int:
    trades = result.trades
    if trades.empty:
        return 0
    dates = pd.to_datetime(trades["Tanggal tutup"], errors="coerce")
    net = pd.to_numeric(trades["Net P/L"], errors="coerce").fillna(0.0)
    quarterly = net.groupby(dates.dt.to_period("Q")).sum()
    return int((quarterly > 0).sum())


def _config_label(config: SignalQualityConfig) -> str:
    values = []
    for key, value in asdict(config).items():
        if key in {"name", "group"} or value in {None, False, 0, 1.0}:
            continue
        values.append(f"{key}={value}")
    return "Entry v1 tanpa filter" if not values else " | ".join(values)


def _development_score(row: pd.Series, baseline: dict[str, object]) -> float:
    baseline_trades = max(float(baseline["Jumlah transaksi"]), 1.0)
    trade_ratio = float(row["Transaksi"]) / baseline_trades
    dd_score = 25 * max(0.0, 1 - float(row["Max drawdown (%)"]) / 20)
    pf_score = 30 * min(float(row["Profit factor"]) / PROFIT_FACTOR_TARGET, 1.0)
    growth_score = 25 * max(0.0, min(float(row["Growth (%)"]) / 50, 1.0))
    retention = float(row["Retensi entry (%)"])
    retention_score = 10 * max(0.0, 1 - abs(retention - 75) / 25)
    stability_score = 10 * min(float(row["Kuartal positif"]) / 4, 1.0)
    sample_score = 10 * min(trade_ratio, 1.0)
    return dd_score + pf_score + growth_score + sample_score + retention_score + stability_score


def _decision_table(
    validation: pd.DataFrame,
    stress: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    monthly: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, candidate in validation[validation["Kelompok"].eq("Balanced v3")].iterrows():
        name = str(candidate["Kandidat"])
        candidate_stress = stress[stress["Kandidat"].eq(name)]
        mc = monte_carlo[monte_carlo["Kandidat"].eq(name)].iloc[0]
        month = monthly[name]
        mc_loss = float(mc["Probabilitas equity akhir < modal awal (%)"])
        criteria = {
            "Growth positif": bool(candidate["Growth (%)"] > 0),
            "Drawdown <= 10%": bool(candidate["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT),
            "Profit factor >= 1.30": bool(candidate["Profit factor"] >= PROFIT_FACTOR_TARGET),
            "Monte Carlo rugi <= 10%": mc_loss <= MAX_MONTE_CARLO_LOSS_PCT,
            "Stress profitable 9/9": bool(
                len(candidate_stress) == 9 and (candidate_stress["Growth (%)"] > 0).all()
            ),
            "Transaksi >= 50": bool(candidate["Transaksi"] >= MIN_TRADES),
        }
        positive_months = int((month["Net P/L"] > 0).sum())
        score = (
            30 * min(max(float(candidate["Profit factor"]), 0) / PROFIT_FACTOR_TARGET, 1)
            + 25 * max(0, 1 - float(candidate["Max drawdown (%)"]) / MAX_DRAWDOWN_PCT)
            + 20 * max(0, 1 - mc_loss / MAX_MONTE_CARLO_LOSS_PCT)
            + 15 * positive_months / max(len(month), 1)
            + 10 * min(max(float(candidate["Growth (%)"]), 0) / 20, 1)
        )
        rows.append(
            {
                "Kandidat": name,
                **criteria,
                "Jumlah kriteria lolos": sum(criteria.values()),
                "Lulus seluruh kriteria": all(criteria.values()),
                "Bulan positif": positive_months,
                "Monte Carlo rugi (%)": mc_loss,
                "Final score": score,
            }
        )
    return pd.DataFrame(rows)
