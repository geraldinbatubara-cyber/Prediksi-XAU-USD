from __future__ import annotations

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import POINT_SIZE, _prepare_m1
from gold_forecast.v1_entry_outcome import (
    _balanced_signals,
    _safe_monthly_summary,
    _safe_monte_carlo,
)
from gold_forecast.v1_entry_quality import _stress_test
from gold_forecast.v1_entry_quality_path import (
    CONFIRMATION_END,
    CONFIRMATION_START,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    FOLDS,
    _data_audit,
    _unique_signals,
)
from gold_forecast.v1_fixed_delay import _build_fixed_delay_signals
from gold_forecast.v1_risk_control import (
    RiskControlConfig,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_signal_quality import SignalQualityConfig, _entry_features
from gold_forecast.v1_unified_benchmark import _unified_best


CALIBRATION_END = pd.Timestamp("2023-12-31 23:59:59")
VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2024-12-31 23:59:59")
DEVELOPMENT_CONFIRMATION_START = pd.Timestamp("2025-01-01")
DEVELOPMENT_CONFIRMATION_END = DEVELOPMENT_END
PROFIT_FACTOR_TARGET = 1.50
MAX_DRAWDOWN_PCT = 10.0
MIN_RETENTION_PCT = 60.0

CANDIDATES = (
    "Fixed Delay 5m Control",
    "Direction Guard",
    "Trend Strength Guard",
    "Volatility Corridor",
    "Spread-to-Opportunity Guard",
    "Direction + Trend Guard",
)


def run_v1_fixed_delay_quality_lab(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = _unified_best(leaderboard.iloc[0].to_dict())
    entry_features = _entry_features(data)
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    balanced_config = SignalQualityConfig(
        "Balanced Entry Frozen",
        "Trend engine",
        conviction_multiplier=1.05,
        require_h1_trend=True,
        wait_hours=2,
    )
    config = RiskControlConfig(
        "Fixed Delay Quality Guard",
        "Entry filter only",
        max_total_positions=1,
        max_same_direction=1,
    )

    development_control, development_events = _fixed_delay_control(
        data,
        signal_daily,
        best,
        entry_features,
        balanced_config,
        spread_limit,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )
    confirmation_control, confirmation_events = _fixed_delay_control(
        data,
        signal_daily,
        best,
        entry_features,
        balanced_config,
        spread_limit,
        CONFIRMATION_START,
        CONFIRMATION_END,
    )
    development_features = _guard_features(
        development_control, development_events, entry_features
    )
    confirmation_features = _guard_features(
        confirmation_control, confirmation_events, entry_features
    )
    thresholds = _calibrate_thresholds(
        development_features.loc[DEVELOPMENT_START:CALIBRATION_END]
    )
    development_signals = _candidate_signals(
        development_control, development_features, thresholds
    )
    confirmation_signals = _candidate_signals(
        confirmation_control, confirmation_features, thresholds
    )

    development_data = data.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    confirmation_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    development_results = _simulate_candidates(
        development_data, development_signals, best, config
    )
    confirmation_results = _simulate_candidates(
        confirmation_data, confirmation_signals, best, config
    )
    development = _result_table(development_results, development_signals)
    confirmation = _result_table(confirmation_results, confirmation_signals)
    period_validation = _period_validation(
        data, development_signals, best, config
    )
    folds = _fold_evaluation(data, development_signals, best, config)
    retention = _retention_table(development_signals, confirmation_signals)
    decisions = _decision_table(
        development, period_validation, folds, retention
    )
    ranking = _ranking_table(development, confirmation, decisions)
    winner = str(ranking.iloc[0]["Kandidat"])
    winner_result = development_results[winner]
    monte_carlo, monte_carlo_summary = _safe_monte_carlo(winner_result.trades)
    stress = _stress_test(
        development_data, development_signals[winner], best, config
    )
    monthly = _safe_monthly_summary(winner_result)

    return {
        "methodology": {
            "Control": "Fixed Delay 5m dari Unified Strategy Benchmark; tidak diubah",
            "Calibration": "01 Jan 2022 - 31 Des 2023; threshold berbasis quantile fitur tanpa label outcome",
            "Validation": "01 Jan 2024 - 31 Des 2024",
            "Development confirmation": "01 Jan 2025 - 31 Des 2025",
            "Historical reference": "01 Jan 2026 - 30 Jun 2026; tidak dipakai memilih kandidat",
            "Execution contract": (
                "Equity USD 1.000 | lot 0.01 | maksimal 1 posisi | TP USD 25 | "
                "SL USD 10 | spread M1 aktual | slippage 2 points/sisi"
            ),
            "Threshold policy": (
                "Direction Q25; trend strength Q25; volatility Q10-Q90; "
                "spread/ATR Q75; kombinasi Direction+Trend memakai Q20 masing-masing"
            ),
            "Baseline lock": (
                "Baseline v1, Fixed Delay paper live, ledger, dan seluruh eksperimen lama tidak diubah"
            ),
        },
        "thresholds": _threshold_table(thresholds),
        "data_audit": _extended_data_audit(data),
        "development": development,
        "period_validation": period_validation,
        "confirmation": confirmation,
        "folds": folds,
        "retention": retention,
        "decisions": decisions,
        "ranking": ranking,
        "winner": winner,
        "winner_stress": stress,
        "winner_monthly": monthly,
        "winner_monte_carlo": monte_carlo,
        "winner_monte_carlo_summary": monte_carlo_summary,
        "signal_audit": _signal_audit(
            development_features, development_signals, thresholds
        ),
    }


def _fixed_delay_control(
    data: pd.DataFrame,
    daily: pd.DataFrame,
    best: dict[str, object],
    entry_features: pd.DataFrame,
    balanced_config: SignalQualityConfig,
    spread_limit: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    balanced = _unique_signals(
        _balanced_signals(
            data,
            daily,
            best,
            entry_features,
            balanced_config,
            spread_limit,
            start,
            end,
        )
    )
    delayed, events = _build_fixed_delay_signals(
        data, balanced, best, 5, spread_limit
    )
    return _unique_signals(delayed), events


def _guard_features(
    signals: pd.DataFrame,
    events: pd.DataFrame,
    entry_features: pd.DataFrame,
) -> pd.DataFrame:
    columns = (
        "direction_move_atr",
        "trend_strength_atr",
        "volatility_pct",
        "spread_to_atr",
    )
    if signals.empty or events.empty:
        return pd.DataFrame(index=signals.index, columns=columns, dtype=float)

    accepted = events.loc[~events["expired"] & events["spread_ok"]].copy()
    accepted["confirmation_time"] = pd.to_datetime(
        accepted["confirmation_time"], errors="coerce"
    )
    accepted = accepted.dropna(subset=["confirmation_time"])
    accepted = accepted.loc[
        ~accepted["confirmation_time"].duplicated(keep="first")
    ].set_index("confirmation_time")
    accepted = accepted.reindex(signals.index)
    aligned = entry_features.reindex(signals.index)
    atr = pd.to_numeric(aligned["atr"], errors="coerce")
    price = pd.to_numeric(aligned["price"], errors="coerce")
    spread = pd.to_numeric(accepted["spread_points"], errors="coerce")

    output = pd.DataFrame(index=signals.index)
    output["direction_move_atr"] = pd.to_numeric(
        accepted["signed_return_atr"], errors="coerce"
    )
    output["trend_strength_atr"] = (
        pd.to_numeric(aligned["h1_fast"], errors="coerce")
        - pd.to_numeric(aligned["h1_slow"], errors="coerce")
    ).abs() / atr
    output["volatility_pct"] = atr / price * 100
    output["spread_to_atr"] = spread * POINT_SIZE / atr
    return output.replace([np.inf, -np.inf], np.nan)


def _calibrate_thresholds(features: pd.DataFrame) -> dict[str, float]:
    clean = features.dropna()
    if clean.empty:
        raise RuntimeError("Fitur calibration Fixed Delay Quality Guard kosong.")
    return {
        "direction_min": float(clean["direction_move_atr"].quantile(0.25)),
        "trend_min": float(clean["trend_strength_atr"].quantile(0.25)),
        "volatility_min": float(clean["volatility_pct"].quantile(0.10)),
        "volatility_max": float(clean["volatility_pct"].quantile(0.90)),
        "spread_to_atr_max": float(clean["spread_to_atr"].quantile(0.75)),
        "combined_direction_min": float(
            clean["direction_move_atr"].quantile(0.20)
        ),
        "combined_trend_min": float(
            clean["trend_strength_atr"].quantile(0.20)
        ),
    }


def _candidate_signals(
    control: pd.DataFrame,
    features: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict[str, pd.DataFrame]:
    valid = features.notna().all(axis=1)
    masks = {
        "Fixed Delay 5m Control": pd.Series(True, index=control.index),
        "Direction Guard": (
            valid
            & features["direction_move_atr"].ge(thresholds["direction_min"])
        ),
        "Trend Strength Guard": (
            valid
            & features["trend_strength_atr"].ge(thresholds["trend_min"])
        ),
        "Volatility Corridor": (
            valid
            & features["volatility_pct"].between(
                thresholds["volatility_min"],
                thresholds["volatility_max"],
                inclusive="both",
            )
        ),
        "Spread-to-Opportunity Guard": (
            valid
            & features["spread_to_atr"].le(
                thresholds["spread_to_atr_max"]
            )
        ),
        "Direction + Trend Guard": (
            valid
            & features["direction_move_atr"].ge(
                thresholds["combined_direction_min"]
            )
            & features["trend_strength_atr"].ge(
                thresholds["combined_trend_min"]
            )
        ),
    }
    return {
        candidate: control.loc[masks[candidate].fillna(False)].copy()
        for candidate in CANDIDATES
    }


def _simulate_candidates(
    data: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    best: dict[str, object],
    config: RiskControlConfig,
) -> dict[str, object]:
    return {
        candidate: _simulate_risk_control(
            data, signals[candidate], best, config
        )
        for candidate in CANDIDATES
    }


def _result_table(
    results: dict[str, object],
    signals: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Kandidat": candidate,
                "Sinyal tersedia": len(signals[candidate]),
                **_metric_values(results[candidate]),
            }
            for candidate in CANDIDATES
        ]
    )


def _period_validation(
    data: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    periods = (
        ("Calibration 2022-2023", DEVELOPMENT_START, CALIBRATION_END),
        ("Validation 2024", VALIDATION_START, VALIDATION_END),
        (
            "Development confirmation 2025",
            DEVELOPMENT_CONFIRMATION_START,
            DEVELOPMENT_CONFIRMATION_END,
        ),
    )
    rows = []
    for label, start, end in periods:
        period_data = data.loc[start:end]
        for candidate in CANDIDATES:
            period_signals = signals[candidate].loc[start:end]
            result = _simulate_risk_control(
                period_data, period_signals, best, config
            )
            rows.append(
                {
                    "Periode": label,
                    "Kandidat": candidate,
                    "Sinyal tersedia": len(period_signals),
                    **_metric_values(result),
                }
            )
    return pd.DataFrame(rows)


def _fold_evaluation(
    data: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        period_data = data.loc[fold.test_start:fold.test_end]
        for candidate in CANDIDATES:
            selected = signals[candidate].loc[fold.test_start:fold.test_end]
            result = _simulate_risk_control(
                period_data, selected, best, config
            )
            metrics = _metric_values(result)
            rows.append(
                {
                    "Fold": fold.name,
                    "Kandidat": candidate,
                    "Test mulai": fold.test_start,
                    "Test akhir": fold.test_end,
                    **metrics,
                    "Profitable": bool(metrics["Growth (%)"] > 0),
                }
            )
    return pd.DataFrame(rows)


def _retention_table(
    development: dict[str, pd.DataFrame],
    confirmation: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    dev_control = max(len(development["Fixed Delay 5m Control"]), 1)
    test_control = max(len(confirmation["Fixed Delay 5m Control"]), 1)
    return pd.DataFrame(
        [
            {
                "Kandidat": candidate,
                "Sinyal development": len(development[candidate]),
                "Retensi development (%)": (
                    len(development[candidate]) / dev_control * 100
                ),
                "Sinyal 2026H1": len(confirmation[candidate]),
                "Retensi 2026H1 (%)": (
                    len(confirmation[candidate]) / test_control * 100
                ),
            }
            for candidate in CANDIDATES
        ]
    )


def _decision_table(
    development: pd.DataFrame,
    period_validation: pd.DataFrame,
    folds: pd.DataFrame,
    retention: pd.DataFrame,
) -> pd.DataFrame:
    dev = development.set_index("Kandidat")
    retained = retention.set_index("Kandidat")
    period = period_validation.set_index(["Periode", "Kandidat"])
    rows = []
    for candidate in CANDIDATES:
        strategy_folds = folds[folds["Kandidat"].eq(candidate)]
        criteria = {
            "Growth development positif": float(
                dev.loc[candidate, "Growth (%)"]
            ) > 0,
            "PF development >= 1.50": float(
                dev.loc[candidate, "Profit factor"]
            ) >= PROFIT_FACTOR_TARGET,
            "DD development <= 10%": float(
                dev.loc[candidate, "Max drawdown (%)"]
            ) <= MAX_DRAWDOWN_PCT,
            "Retensi transaksi >= 60%": float(
                retained.loc[candidate, "Retensi development (%)"]
            ) >= MIN_RETENTION_PCT,
            "Validation 2024 positif": float(
                period.loc[("Validation 2024", candidate), "Growth (%)"]
            ) > 0,
            "Confirmation 2025 positif": float(
                period.loc[
                    ("Development confirmation 2025", candidate),
                    "Growth (%)",
                ]
            ) > 0,
            "Fold profitable >= 9/12": int(
                strategy_folds["Profitable"].sum()
            ) >= 9,
        }
        rows.append(
            {
                "Kandidat": candidate,
                **{name: bool(value) for name, value in criteria.items()},
                "Fold profitable": int(strategy_folds["Profitable"].sum()),
                "Kriteria lolos": int(sum(criteria.values())),
                "Total kriteria": len(criteria),
                "Lulus": bool(all(criteria.values())),
            }
        )
    return pd.DataFrame(rows)


def _ranking_table(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    dev = development.set_index("Kandidat")
    test = confirmation.set_index("Kandidat")
    decision = decisions.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        rows.append(
            {
                "Kandidat": candidate,
                "Growth development (%)": float(
                    dev.loc[candidate, "Growth (%)"]
                ),
                "PF development": float(
                    dev.loc[candidate, "Profit factor"]
                ),
                "DD development (%)": float(
                    dev.loc[candidate, "Max drawdown (%)"]
                ),
                "Transaksi development": int(
                    dev.loc[candidate, "Transaksi"]
                ),
                "Growth 2026H1 (%)": float(
                    test.loc[candidate, "Growth (%)"]
                ),
                "PF 2026H1": float(test.loc[candidate, "Profit factor"]),
                "DD 2026H1 (%)": float(
                    test.loc[candidate, "Max drawdown (%)"]
                ),
                "Transaksi 2026H1": int(
                    test.loc[candidate, "Transaksi"]
                ),
                "Fold profitable": int(
                    decision.loc[candidate, "Fold profitable"]
                ),
                "Kriteria lolos": int(
                    decision.loc[candidate, "Kriteria lolos"]
                ),
                "Lulus": bool(decision.loc[candidate, "Lulus"]),
            }
        )
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        [
            "Lulus",
            "Kriteria lolos",
            "PF development",
            "DD development (%)",
            "Growth development (%)",
        ],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _threshold_table(thresholds: dict[str, float]) -> pd.DataFrame:
    labels = {
        "direction_min": "Direction move minimum (ATR)",
        "trend_min": "Trend strength minimum (ATR)",
        "volatility_min": "Volatilitas minimum (%)",
        "volatility_max": "Volatilitas maksimum (%)",
        "spread_to_atr_max": "Spread/ATR maksimum",
        "combined_direction_min": "Kombinasi: direction minimum (ATR)",
        "combined_trend_min": "Kombinasi: trend minimum (ATR)",
    }
    return pd.DataFrame(
        [
            {"Parameter": labels[key], "Nilai": value}
            for key, value in thresholds.items()
        ]
    )


def _signal_audit(
    features: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    control_count = max(len(signals["Fixed Delay 5m Control"]), 1)
    for candidate in CANDIDATES:
        rows.append(
            {
                "Kandidat": candidate,
                "Sinyal diterima": len(signals[candidate]),
                "Sinyal ditolak": control_count - len(signals[candidate]),
                "Retensi (%)": len(signals[candidate]) / control_count * 100,
            }
        )
    rows.append(
        {
            "Kandidat": "Audit fitur lengkap",
            "Sinyal diterima": int(features.notna().all(axis=1).sum()),
            "Sinyal ditolak": int(features.isna().any(axis=1).sum()),
            "Retensi (%)": (
                features.notna().all(axis=1).mean() * 100
                if not features.empty
                else 0.0
            ),
        }
    )
    return pd.DataFrame(rows)


def _extended_data_audit(data: pd.DataFrame) -> pd.DataFrame:
    audit = _data_audit(data)
    expected = set(pd.period_range("2022-01", "2026-06", freq="M"))
    actual = set(
        data.loc[DEVELOPMENT_START:CONFIRMATION_END]
        .index.to_period("M")
        .unique()
    )
    coverage = audit["Pemeriksaan"].str.startswith("Cakupan bulan")
    audit.loc[coverage, "Pemeriksaan"] = (
        "Cakupan bulan 2022-01 sampai 2026-06"
    )
    audit.loc[coverage, "Status"] = (
        "LOLOS" if expected.issubset(actual) else "BELUM"
    )
    audit.loc[coverage, "Detail"] = (
        f"{len(expected & actual)}/{len(expected)} bulan tersedia"
    )
    return audit
