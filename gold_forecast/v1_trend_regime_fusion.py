from __future__ import annotations

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import _prepare_m1
from gold_forecast.v1_entry_outcome import (
    _balanced_signals,
    _safe_monte_carlo,
)
from gold_forecast.v1_entry_quality import _stress_test
from gold_forecast.v1_entry_quality_path import (
    CONFIRMATION_END,
    CONFIRMATION_START,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    FOLDS,
    _unique_signals,
)
from gold_forecast.v1_fixed_delay import _build_fixed_delay_signals
from gold_forecast.v1_regime_classifier import (
    FEATURE_COLUMNS,
    _classification_metrics,
    _classifier_frame,
    _state_machine,
)
from gold_forecast.v1_risk_control import (
    RiskControlConfig,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_signal_quality import SignalQualityConfig, _entry_features
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


CALIBRATION_END = pd.Timestamp("2023-12-31 23:59:59")
VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2024-12-31 23:59:59")
DEVELOPMENT_CONFIRMATION_START = pd.Timestamp("2025-01-01")
DEVELOPMENT_CONFIRMATION_END = DEVELOPMENT_END
Q30_THRESHOLD = 0.597140
Q40_THRESHOLD = 0.736217
PROFIT_FACTOR_TARGET = 1.50
MAX_DRAWDOWN_PCT = 10.0
MIN_RETENTION_PCT = 60.0
MAX_MONTE_CARLO_LOSS_PCT = 10.0

CANDIDATES = (
    "Fixed Delay Control",
    "Regime Gate Only",
    "Q40 Only",
    "Fusion Q30",
    "Fusion Q40",
    "Fusion Q40 Confirmed",
)


def run_v1_trend_regime_fusion_lab(
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
    simulation_config = RiskControlConfig(
        "Trend-Regime Fusion",
        "Ablation",
        max_total_positions=1,
        max_same_direction=1,
    )

    classifier = _classifier_frame(data)
    calibration = classifier.loc[DEVELOPMENT_START:CALIBRATION_END]
    calibration = calibration.dropna(subset=FEATURE_COLUMNS)
    thresholds = _calibrate_regime_thresholds(calibration)
    usable = classifier.dropna(subset=FEATURE_COLUMNS)
    probabilities = _calibrated_rule_probabilities(usable, thresholds)
    states = _state_machine(probabilities, usable)
    state_m1 = states.reindex(data.index, method="ffill").fillna("UNCERTAIN")

    balanced = _unique_signals(
        _balanced_signals(
            data,
            signal_daily,
            best,
            entry_features,
            balanced_config,
            spread_limit,
            DEVELOPMENT_START,
            CONFIRMATION_END,
        )
    )
    candidate_inputs, input_audit = _candidate_inputs(
        balanced, entry_features, state_m1
    )
    signals, delay_audit = _delay_candidates(
        data, candidate_inputs, state_m1, best, spread_limit
    )

    development_data = data.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    reference_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    development_results = _simulate_all(
        development_data, signals, best, simulation_config,
        DEVELOPMENT_START, DEVELOPMENT_END,
    )
    reference_results = _simulate_all(
        reference_data, signals, best, simulation_config,
        CONFIRMATION_START, CONFIRMATION_END,
    )
    development = _result_table(development_results, signals, DEVELOPMENT_START, DEVELOPMENT_END)
    reference = _result_table(reference_results, signals, CONFIRMATION_START, CONFIRMATION_END)
    periods = _period_validation(data, signals, best, simulation_config)
    folds = _fold_evaluation(data, signals, best, simulation_config)
    monte_carlo = _monte_carlo_summary(development_results)
    direction = _direction_audit(
        development_results, reference_results
    )
    retention = _retention_table(signals)
    stress_candidates = _stress_candidates(
        development, periods, folds, retention, monte_carlo, direction
    )
    stress = _stress_summary(
        development_data,
        signals,
        best,
        simulation_config,
        stress_candidates,
    )
    decisions = _decision_table(
        development, periods, folds, retention, monte_carlo, stress, direction
    )
    ranking = _ranking_table(development, reference, retention, decisions)
    rejected = _rejected_signal_audit(
        development_results, reference_results
    )
    classifier_validation = _classifier_validation(classifier, states)

    return {
        "methodology": {
            "Question": (
                "Apakah Regime Classifier v2 dan Trend Strength dapat meningkatkan kualitas "
                "Fixed Delay 5m tanpa menghilangkan produktivitas?"
            ),
            "Calibration": (
                "2022-2023; threshold classifier dihitung hanya dari distribusi fitur, "
                "tanpa label profit"
            ),
            "Validation": "2024",
            "Development confirmation": "2025",
            "Historical reference": (
                "2026H1 hanya referensi historis dan tidak dipakai memilih kandidat"
            ),
            "Architecture": (
                "Baseline v1 -> Regime Classifier v2 -> Trend Strength -> "
                "Balanced Entry -> Fixed Delay 5m -> Entry"
            ),
            "Regime rule": (
                "TREND_UP wajib selaras BUY dan TREND_DOWN wajib selaras SELL"
            ),
            "Execution contract": (
                "Equity USD 1.000 | lot 0.01 | TP USD 25 | SL USD 10 | "
                "maksimal 1 posisi | spread M1 aktual | slippage 2 points/sisi"
            ),
            "Baseline lock": (
                "Baseline v1, paper live, ledger, dan parameter observasi tidak diubah"
            ),
        },
        "thresholds": _threshold_table(thresholds),
        "classifier_validation": classifier_validation,
        "data_audit": _extended_data_audit(data),
        "input_audit": input_audit,
        "delay_audit": delay_audit,
        "development": development,
        "period_validation": periods,
        "historical_reference": reference,
        "folds": folds,
        "retention": retention,
        "monte_carlo_summary": monte_carlo,
        "stress_summary": stress,
        "direction_audit": direction,
        "rejected_signal_audit": rejected,
        "decisions": decisions,
        "ranking": ranking,
        "winner": str(ranking.iloc[0]["Kandidat"]),
    }


def _calibrate_regime_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        raise RuntimeError("Fitur calibration Regime Classifier kosong.")
    return {
        "adx_max": float(frame["adx"].quantile(0.35)),
        "efficiency_max": float(frame["efficiency"].quantile(0.35)),
        "choppiness_min": float(frame["choppiness"].quantile(0.65)),
        "gap_max": float(frame["ema_gap_atr"].abs().quantile(0.35)),
        "slope_max": float(frame["ema_slow_slope_atr"].abs().quantile(0.35)),
    }


def _calibrated_rule_probabilities(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    output = pd.DataFrame(
        0.05,
        index=frame.index,
        columns=("TREND_DOWN", "SIDEWAYS", "TRANSITION", "TREND_UP"),
    )
    side_votes = (
        frame["adx"].le(thresholds["adx_max"]).astype(int)
        + frame["efficiency"].le(thresholds["efficiency_max"]).astype(int)
        + frame["choppiness"].ge(thresholds["choppiness_min"]).astype(int)
        + frame["ema_gap_atr"].abs().le(thresholds["gap_max"]).astype(int)
        + frame["ema_slow_slope_atr"].abs().le(thresholds["slope_max"]).astype(int)
    )
    up_score = (
        frame["ema_gap_atr"].gt(0).astype(float)
        + frame["return_3"].gt(0).astype(float)
        + frame["h4_gap_atr"].gt(0).astype(float)
        + frame["breakout_up"]
    ) / 4
    down_score = (
        frame["ema_gap_atr"].lt(0).astype(float)
        + frame["return_3"].lt(0).astype(float)
        + frame["h4_gap_atr"].lt(0).astype(float)
        + frame["breakout_down"]
    ) / 4
    output["SIDEWAYS"] = 0.10 + 0.14 * side_votes
    output["TREND_UP"] = 0.10 + 0.55 * up_score * (1 - side_votes / 5)
    output["TREND_DOWN"] = 0.10 + 0.55 * down_score * (1 - side_votes / 5)
    output["TRANSITION"] = 0.20 + 0.10 * side_votes.between(2, 3).astype(float)
    return output.div(output.sum(axis=1), axis=0)


def _candidate_inputs(
    balanced: pd.DataFrame,
    features: pd.DataFrame,
    states: pd.Series,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    expected = pd.to_numeric(balanced["expected_change_pct"], errors="coerce")
    direction = pd.Series(
        np.where(expected.gt(0), "BUY", "SELL"), index=balanced.index
    )
    state = states.reindex(balanced.index, method="ffill").fillna("UNCERTAIN")
    aligned_regime = (
        direction.eq("BUY") & state.eq("TREND_UP")
    ) | (
        direction.eq("SELL") & state.eq("TREND_DOWN")
    )
    aligned = features.reindex(balanced.index)
    strength = (
        pd.to_numeric(aligned["h1_fast"], errors="coerce")
        - pd.to_numeric(aligned["h1_slow"], errors="coerce")
    ).abs() / pd.to_numeric(aligned["atr"], errors="coerce").replace(0, np.nan)
    masks = {
        "Fixed Delay Control": pd.Series(True, index=balanced.index),
        "Regime Gate Only": aligned_regime,
        "Q40 Only": strength.ge(Q40_THRESHOLD),
        "Fusion Q30": aligned_regime & strength.ge(Q30_THRESHOLD),
        "Fusion Q40": aligned_regime & strength.ge(Q40_THRESHOLD),
        "Fusion Q40 Confirmed": aligned_regime & strength.ge(Q40_THRESHOLD),
    }
    candidates = {
        name: balanced.loc[mask.fillna(False)].copy()
        for name, mask in masks.items()
    }
    audit = pd.DataFrame(
        [
            {
                "Lapisan": name,
                "Sinyal masuk": len(balanced),
                "Sinyal lolos sebelum delay": len(candidates[name]),
                "Retensi sebelum delay (%)": len(candidates[name]) / max(len(balanced), 1) * 100,
            }
            for name in CANDIDATES
        ]
    )
    return candidates, audit


def _delay_candidates(
    data: pd.DataFrame,
    candidate_inputs: dict[str, pd.DataFrame],
    states: pd.Series,
    best: dict[str, object],
    spread_limit: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    output = {}
    rows = []
    for candidate in CANDIDATES:
        delayed, events = _build_fixed_delay_signals(
            data, candidate_inputs[candidate], best, 5, spread_limit
        )
        before_confirmation = len(delayed)
        if candidate == "Fusion Q40 Confirmed" and not delayed.empty:
            expected = pd.to_numeric(delayed["expected_change_pct"], errors="coerce")
            direction = pd.Series(
                np.where(expected.gt(0), "BUY", "SELL"), index=delayed.index
            )
            delayed_state = states.reindex(delayed.index, method="ffill").fillna("UNCERTAIN")
            confirmed = (
                direction.eq("BUY") & delayed_state.eq("TREND_UP")
            ) | (
                direction.eq("SELL") & delayed_state.eq("TREND_DOWN")
            )
            delayed = delayed.loc[confirmed].copy()
        output[candidate] = _unique_signals(delayed)
        rows.append(
            {
                "Kandidat": candidate,
                "Sinyal sebelum delay": len(candidate_inputs[candidate]),
                "Lolos barrier dan spread": before_confirmation,
                "Lolos konfirmasi regime kedua": len(output[candidate]),
                "Batal barrier": int(events["expired"].sum()) if not events.empty else 0,
                "Batal spread": int(
                    (~events["spread_ok"] & ~events["expired"]).sum()
                ) if not events.empty else 0,
            }
        )
    return output, pd.DataFrame(rows)


def _simulate_all(data, signals, best, config, start, end):
    return {
        name: _simulate_risk_control(data, frame.loc[start:end], best, config)
        for name, frame in signals.items()
    }


def _result_table(results, signals, start, end):
    return pd.DataFrame(
        [
            {
                "Kandidat": name,
                "Sinyal tersedia": len(signals[name].loc[start:end]),
                **_metric_values(results[name]),
            }
            for name in CANDIDATES
        ]
    )


def _period_validation(data, signals, best, config):
    periods = (
        ("Calibration diagnostic 2022-2023", DEVELOPMENT_START, CALIBRATION_END),
        ("Validation 2024", VALIDATION_START, VALIDATION_END),
        ("Development confirmation 2025", DEVELOPMENT_CONFIRMATION_START, DEVELOPMENT_CONFIRMATION_END),
    )
    rows = []
    for label, start, end in periods:
        period_data = data.loc[start:end]
        for candidate in CANDIDATES:
            selected = signals[candidate].loc[start:end]
            result = _simulate_risk_control(period_data, selected, best, config)
            rows.append({
                "Periode": label,
                "Kandidat": candidate,
                "Sinyal tersedia": len(selected),
                **_metric_values(result),
            })
    return pd.DataFrame(rows)


def _fold_evaluation(data, signals, best, config):
    rows = []
    for fold in FOLDS:
        period_data = data.loc[fold.test_start:fold.test_end]
        for candidate in CANDIDATES:
            result = _simulate_risk_control(
                period_data,
                signals[candidate].loc[fold.test_start:fold.test_end],
                best,
                config,
            )
            metrics = _metric_values(result)
            rows.append({
                "Fold": fold.name,
                "Kelompok": "Calibration diagnostic" if fold.test_start.year == 2023 else "Primary validation",
                "Kandidat": candidate,
                "Test mulai": fold.test_start,
                "Test akhir": fold.test_end,
                **metrics,
                "Profitable": bool(metrics["Growth (%)"] > 0),
            })
    return pd.DataFrame(rows)


def _monte_carlo_summary(results):
    rows = []
    for candidate in CANDIDATES:
        _, summary = _safe_monte_carlo(results[candidate].trades)
        rows.append({"Kandidat": candidate, **summary})
    return pd.DataFrame(rows)


def _stress_candidates(development, periods, folds, retention, monte_carlo, direction):
    dev = development.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    retained = retention.set_index("Kandidat")
    mc = monte_carlo.set_index("Kandidat")
    selected = []
    for candidate in CANDIDATES:
        primary = folds[
            folds["Kandidat"].eq(candidate)
            & folds["Kelompok"].eq("Primary validation")
        ]
        candidate_direction = direction[
            direction["Kandidat"].eq(candidate)
            & direction["Periode"].eq("Development 2022-2025")
        ]
        total = max(float(candidate_direction["Transaksi"].sum()), 1.0)
        minor_share = float(candidate_direction["Transaksi"].min() / total * 100)
        core = (
            float(dev.loc[candidate, "Growth (%)"]) > 0
            and float(retained.loc[candidate, "Retensi development (%)"]) >= MIN_RETENTION_PCT
            and float(period.loc[("Validation 2024", candidate), "Growth (%)"]) > 0
            and float(period.loc[("Development confirmation 2025", candidate), "Growth (%)"]) > 0
            and int(primary["Profitable"].sum()) >= 6
            and float(
                mc.loc[
                    candidate,
                    "Probabilitas equity akhir < modal awal (%)",
                ]
            ) <= MAX_MONTE_CARLO_LOSS_PCT
            and minor_share >= 15.0
        )
        if core:
            selected.append(candidate)
    return selected


def _stress_summary(data, signals, best, config, selected_candidates):
    rows = []
    for candidate in CANDIDATES:
        if candidate not in selected_candidates:
            rows.append({
                "Kandidat": candidate,
                "Skenario profitable": 0,
                "Jumlah skenario": 9,
                "Worst growth (%)": np.nan,
                "Worst drawdown (%)": np.nan,
                "Status": "TIDAK DIUJI - gagal gerbang dasar",
            })
            continue
        stress = _stress_test(
            data,
            signals[candidate].loc[data.index.min():data.index.max()],
            best,
            config,
        )
        rows.append({
            "Kandidat": candidate,
            "Skenario profitable": int(stress["Growth (%)"].gt(0).sum()),
            "Jumlah skenario": len(stress),
            "Worst growth (%)": float(stress["Growth (%)"].min()),
            "Worst drawdown (%)": float(stress["Max drawdown (%)"].max()),
            "Status": "DIUJI",
        })
    return pd.DataFrame(rows)


def _direction_audit(development_results, reference_results):
    rows = []
    for period, results in (
        ("Development 2022-2025", development_results),
        ("Historical reference 2026H1", reference_results),
    ):
        for candidate in CANDIDATES:
            trades = results[candidate].trades
            for direction in ("BUY", "SELL"):
                selected = trades[
                    trades.get(
                        "Arah",
                        pd.Series(index=trades.index, dtype=object),
                    ).eq(direction)
                ]
                rows.append({
                    "Periode": period,
                    "Kandidat": candidate,
                    "Arah": direction,
                    **_trade_metrics(selected),
                })
    return pd.DataFrame(rows)


def _retention_table(signals):
    control_dev = max(len(signals[CANDIDATES[0]].loc[DEVELOPMENT_START:DEVELOPMENT_END]), 1)
    control_test = max(len(signals[CANDIDATES[0]].loc[CONFIRMATION_START:CONFIRMATION_END]), 1)
    return pd.DataFrame([
        {
            "Kandidat": candidate,
            "Sinyal development": len(signals[candidate].loc[DEVELOPMENT_START:DEVELOPMENT_END]),
            "Retensi development (%)": len(signals[candidate].loc[DEVELOPMENT_START:DEVELOPMENT_END]) / control_dev * 100,
            "Sinyal 2026H1": len(signals[candidate].loc[CONFIRMATION_START:CONFIRMATION_END]),
            "Retensi 2026H1 (%)": len(signals[candidate].loc[CONFIRMATION_START:CONFIRMATION_END]) / control_test * 100,
        }
        for candidate in CANDIDATES
    ])


def _decision_table(development, periods, folds, retention, monte_carlo, stress, direction):
    dev = development.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    retained = retention.set_index("Kandidat")
    mc = monte_carlo.set_index("Kandidat")
    stressed = stress.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        primary_folds = folds[
            folds["Kandidat"].eq(candidate) & folds["Kelompok"].eq("Primary validation")
        ]
        candidate_direction = direction[
            direction["Kandidat"].eq(candidate)
            & direction["Periode"].eq("Development 2022-2025")
        ]
        direction_counts = candidate_direction.set_index("Arah")["Transaksi"]
        total_direction = max(float(direction_counts.sum()), 1.0)
        min_direction_share = float(direction_counts.min() / total_direction * 100)
        criteria = {
            "Growth development positif": float(dev.loc[candidate, "Growth (%)"]) > 0,
            "PF development >= 1.50": float(dev.loc[candidate, "Profit factor"]) >= PROFIT_FACTOR_TARGET,
            "DD development <= 10%": float(dev.loc[candidate, "Max drawdown (%)"]) <= MAX_DRAWDOWN_PCT,
            "Retensi >= 60%": float(retained.loc[candidate, "Retensi development (%)"]) >= MIN_RETENTION_PCT,
            "Validation 2024 positif": float(period.loc[("Validation 2024", candidate), "Growth (%)"]) > 0,
            "Confirmation 2025 positif": float(period.loc[("Development confirmation 2025", candidate), "Growth (%)"]) > 0,
            "Primary fold profitable >= 6/8": int(primary_folds["Profitable"].sum()) >= 6,
            "Monte Carlo rugi <= 10%": float(mc.loc[candidate, "Probabilitas equity akhir < modal awal (%)"]) <= MAX_MONTE_CARLO_LOSS_PCT,
            "Stress profitable 9/9": int(stressed.loc[candidate, "Skenario profitable"]) == int(stressed.loc[candidate, "Jumlah skenario"]),
            "Arah minor >= 15% transaksi": min_direction_share >= 15.0,
        }
        rows.append({
            "Kandidat": candidate,
            **{name: bool(value) for name, value in criteria.items()},
            "Primary fold profitable": int(primary_folds["Profitable"].sum()),
            "Porsi arah minor (%)": min_direction_share,
            "Kriteria lolos": int(sum(criteria.values())),
            "Total kriteria": len(criteria),
            "Lulus": bool(all(criteria.values())),
        })
    return pd.DataFrame(rows)


def _ranking_table(development, reference, retention, decisions):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    retained = retention.set_index("Kandidat")
    decision = decisions.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        rows.append({
            "Kandidat": candidate,
            "Growth development (%)": float(dev.loc[candidate, "Growth (%)"]),
            "PF development": float(dev.loc[candidate, "Profit factor"]),
            "DD development (%)": float(dev.loc[candidate, "Max drawdown (%)"]),
            "Transaksi development": int(dev.loc[candidate, "Transaksi"]),
            "Retensi development (%)": float(retained.loc[candidate, "Retensi development (%)"]),
            "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
            "PF 2026H1": float(ref.loc[candidate, "Profit factor"]),
            "DD 2026H1 (%)": float(ref.loc[candidate, "Max drawdown (%)"]),
            "Transaksi 2026H1": int(ref.loc[candidate, "Transaksi"]),
            "Primary fold profitable": int(decision.loc[candidate, "Primary fold profitable"]),
            "Kriteria lolos": int(decision.loc[candidate, "Kriteria lolos"]),
            "Lulus": bool(decision.loc[candidate, "Lulus"]),
        })
    ranking = pd.DataFrame(rows).sort_values(
        ["Lulus", "Kriteria lolos", "PF development", "DD development (%)", "Growth development (%)"],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _rejected_signal_audit(development_results, reference_results):
    rows = []
    for period, results in (
        ("Development 2022-2025", development_results),
        ("Historical reference 2026H1", reference_results),
    ):
        control_trades = results[CANDIDATES[0]].trades
        control_times = pd.to_datetime(
            control_trades.get("Tanggal entry"), errors="coerce"
        )
        for candidate in CANDIDATES[1:]:
            accepted = results[candidate].trades
            accepted_times = set(
                pd.to_datetime(
                    accepted.get("Tanggal entry"), errors="coerce"
                ).dropna()
            )
            rejected = control_trades.loc[~control_times.isin(accepted_times)]
            for status, frame in (("DITERIMA", accepted), ("DITOLAK", rejected)):
                rows.append({
                    "Periode": period,
                    "Kandidat": candidate,
                    "Status": status,
                    **_trade_metrics(frame),
                })
    return pd.DataFrame(rows)


def _trade_metrics(trades):
    if trades.empty:
        return {
            "Transaksi": 0,
            "Net P/L": 0.0,
            "Profit factor": np.nan,
            "Win rate (%)": np.nan,
        }
    net = pd.to_numeric(trades["Net P/L"], errors="coerce").fillna(0.0)
    profit = float(net[net > 0].sum())
    loss = float(-net[net < 0].sum())
    return {
        "Transaksi": int(len(trades)),
        "Net P/L": float(net.sum()),
        "Profit factor": profit / loss if loss > 0 else np.inf,
        "Win rate (%)": float(net.gt(0).mean() * 100),
    }


def _classifier_validation(frame, states):
    rows = []
    for label, start, end in (
        ("Calibration diagnostic 2022-2023", DEVELOPMENT_START, CALIBRATION_END),
        ("Validation 2024", VALIDATION_START, VALIDATION_END),
        ("Development confirmation 2025", DEVELOPMENT_CONFIRMATION_START, DEVELOPMENT_CONFIRMATION_END),
        ("Historical reference 2026H1", CONFIRMATION_START, CONFIRMATION_END),
    ):
        usable = frame.loc[start:end].dropna(subset=["label"])
        prediction = states.reindex(usable.index).fillna("UNCERTAIN")
        rows.append({"Periode": label, **_classification_metrics(usable["label"], prediction)})
    return pd.DataFrame(rows)


def _threshold_table(thresholds):
    labels = {
        "adx_max": "ADX <= quantile 35%",
        "efficiency_max": "Efficiency <= quantile 35%",
        "choppiness_min": "Choppiness >= quantile 65%",
        "gap_max": "|EMA gap / ATR| <= quantile 35%",
        "slope_max": "|EMA slow slope / ATR| <= quantile 35%",
    }
    rows = [
        {"Parameter classifier": labels[key], "Nilai beku": value}
        for key, value in thresholds.items()
    ]
    rows.extend([
        {"Parameter classifier": "Trend Strength Q30 minimum", "Nilai beku": Q30_THRESHOLD},
        {"Parameter classifier": "Trend Strength Q40 minimum", "Nilai beku": Q40_THRESHOLD},
    ])
    return pd.DataFrame(rows)
