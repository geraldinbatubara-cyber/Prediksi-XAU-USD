from __future__ import annotations

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import _prepare_m1
from gold_forecast.v1_directional_specialization import (
    _ledger_metric_values,
    _monte_carlo_summary,
    _trades_in_period,
)
from gold_forecast.v1_risk_control import _metric_values
from gold_forecast.v1_sell_specialist import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    LOCKED_END,
    LOCKED_START,
    REFERENCE_END,
    REFERENCE_START,
    SELECTION_END,
    SELECTION_START,
)
from gold_forecast.v1_sideways_defense import (
    _classifier_metrics,
    _regime_candidates,
    _regime_features,
    _regime_states,
)
from gold_forecast.v1_sideways_specialist import (
    _mean_reversion_opportunities,
    _opportunities_to_signals,
    _profit_concentration,
    _range_quality_frame,
    _train_outcome_model,
)
from gold_forecast.v1_sideways_specialist_v2 import (
    _augment_opportunities,
    _candidate_signals as _v2_candidate_signals,
    _directional_thresholds,
    _session_selection,
    _train_binary_model,
)
from gold_forecast.v1_sideways_specialist_v3 import (
    _build_position_states,
    _enrich_signals,
    _fold_evaluation,
    _periods,
    _state_audit,
    _state_classification_tables,
    _train_state_model,
)
from gold_forecast.v1_sideways_specialist_v5 import _m15_atr
from gold_forecast.v1_sideways_specialist_v5_extensions import (
    ExitPolicy,
    _simulate_policy,
)
from gold_forecast.v1_sideways_specialist_v7 import (
    CONTROL as V7_CONTROL,
    _candidate_signals as _v7_candidate_signals,
    _entry_filter_frame as _v7_entry_filter_frame,
)
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


CONTROL = "v7 Persistence 2xM15 Control"
ADAPTIVE_BOUNDARY = "Adaptive Boundary"
REJECTION_CONFIRMATION = "Rejection Confirmation"
BREAKOUT_HAZARD = "Breakout Hazard Gate"
BOUNDARY_HAZARD = "Adaptive Boundary + Hazard"
FULL_V8 = "Full Sideways v8"
CANDIDATES = (
    CONTROL,
    ADAPTIVE_BOUNDARY,
    REJECTION_CONFIRMATION,
    BREAKOUT_HAZARD,
    BOUNDARY_HAZARD,
    FULL_V8,
)


def run_v1_sideways_specialist_v8(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
    v7_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = {
        **_unified_best(leaderboard.iloc[0].to_dict()),
        "Close-all target equity": False,
        "Max BUY": 1,
        "Max SELL": 1,
    }
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    features, h1, m15 = _regime_features(data)
    range_frame = _range_quality_frame(features, h1)
    opportunities = _augment_opportunities(
        data,
        range_frame,
        _mean_reversion_opportunities(data, range_frame, m15, spread_limit),
    )

    outcome_model, outcome_selection = _train_outcome_model(opportunities)
    persistence_model = _train_binary_model(
        opportunities, "persistence_12h", "Range Persistence", 101
    )
    hazard_model = _train_binary_model(
        opportunities, "adverse_breakout_6h", "Adverse Breakout Hazard", 111
    )
    directional_thresholds, directional_audit = _directional_thresholds(
        opportunities, outcome_model["probability"]
    )
    selected_sessions, session_audit = _session_selection(opportunities)
    v2_signals, _ = _v2_candidate_signals(
        opportunities,
        outcome_model,
        persistence_model,
        hazard_model,
        directional_thresholds,
        selected_sessions,
        best,
    )
    hazard_control = _enrich_from_opportunities(
        v2_signals[BREAKOUT_HAZARD], opportunities
    )

    regime_candidates = _regime_candidates(
        features,
        data.loc[DEVELOPMENT_START:pd.Timestamp("2023-12-31 23:59:59")],
    )
    classifier_development = pd.DataFrame(
        [
            _classifier_metrics(
                features,
                config,
                DEVELOPMENT_START,
                pd.Timestamp("2023-12-31 23:59:59"),
            )
            for config in regime_candidates
        ]
    )
    selected_name = str(
        classifier_development.sort_values(
            ["Macro F1", "Sideways precision", "Balanced accuracy"],
            ascending=False,
        ).iloc[0]["Classifier"]
    )
    selected_regime = next(
        config for config in regime_candidates if config.name == selected_name
    )
    states = _regime_states(features, selected_regime)
    v7_frame, _ = _v7_entry_filter_frame(
        data, features, states, selected_regime, hazard_control
    )
    exact_v7_control = _v7_candidate_signals(hazard_control, v7_frame)[V7_CONTROL]

    thresholds = _adaptive_thresholds(opportunities)
    signal_map, filter_audit = _candidate_signal_map(
        opportunities,
        exact_v7_control,
        hazard_control,
        best,
        thresholds,
    )
    union_signals = pd.concat(signal_map.values()).sort_index()
    union_signals = union_signals.loc[~union_signals.index.duplicated(keep="first")]
    position_states = _build_position_states(data, range_frame, union_signals)
    model_1h = _train_state_model(
        position_states, "adverse_before_tp_1h", "1 jam", 121
    )
    model_3h = _train_state_model(
        position_states, "adverse_before_tp_3h", "3 jam", 131
    )
    position_states["hazard_1h"] = model_1h["probability"]
    position_states["hazard_3h"] = model_3h["probability"]
    atr_m15 = _m15_atr(data)

    development_results = _simulate_candidates(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END],
        signal_map,
        position_states,
        atr_m15,
        model_1h,
        model_3h,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )
    reference_results = _simulate_candidates(
        data.loc[REFERENCE_START:REFERENCE_END],
        signal_map,
        position_states,
        atr_m15,
        model_1h,
        model_3h,
        REFERENCE_START,
        REFERENCE_END,
    )
    development = _result_table(development_results, signal_map)
    reference = _result_table(reference_results, signal_map)
    periods = _period_validation(development_results, signal_map)
    folds = _fold_evaluation(development_results)
    monte_carlo = _monte_carlo_summary(development_results)
    concentration = _profit_concentration(development_results)
    decisions = _decision_table(
        development, reference, periods, folds, monte_carlo, concentration
    )
    ranking = _selection_ranking(
        development, reference, periods, decisions, filter_audit
    )
    eligible = ranking.loc[ranking["Selection eligible"]]
    winner = str(eligible.iloc[0]["Kandidat"]) if not eligible.empty else ""
    winner_passed = bool(
        winner
        and decisions.set_index("Kandidat").loc[winner, "Lulus"]
    )
    classification = _state_classification_tables(
        position_states, model_1h, model_3h
    )

    return {
        "methodology": {
            "Name": "v1 Sideways Specialist v8 - Adaptive Range Engine",
            "Mandat": "Mean reversion BUY di bawah range dan SELL di atas range.",
            "Control": "Control v7 dipertahankan persis sebagai pembanding.",
            "Adaptive boundary": (
                "Batas edge, lebar range, drift midpoint, perubahan lebar, dan "
                "ATR dikalibrasi hanya dari 2022-2023."
            ),
            "Rejection": (
                "Konfirmasi memakai kekuatan reversal M15 relatif terhadap ATR; "
                "tidak memakai data candle masa depan."
            ),
            "Hazard": (
                "Breakout Hazard Gate v2 tetap menjadi proteksi utama. Full v8 "
                "memakai skor boundary/rejection/RR agar filter tidak menjadi nol."
            ),
            "Execution": (
                "Lot 0.01, spread/slippage broker, swap BUY, SELL swap nol, "
                "TP midpoint, SL boundary, time stop 12 jam, satu posisi, BE 1R."
            ),
            "Development": "2022-2025; classifier dan threshold tidak melihat 2024-2026.",
            "Selection": "2024 saja.",
            "Locked confirmation": "2025.",
            "Historical reference": "2026H1; tidak memilih pemenang.",
            "Live trading lock": "BUY Specialist v4 dan ledger paper live tidak diubah.",
        },
        "adaptive_thresholds": thresholds,
        "selected_regime_config": selected_regime.__dict__,
        "classifier_development": classifier_development,
        "outcome_model_selection": outcome_selection,
        "persistence_model_selection": persistence_model["selection"],
        "hazard_model_selection": hazard_model["selection"],
        "directional_thresholds": directional_audit,
        "session_audit": session_audit,
        "filter_audit": filter_audit,
        "development": development,
        "period_validation": periods,
        "historical_reference": reference,
        "folds": folds,
        "monte_carlo_summary": monte_carlo,
        "profit_concentration": concentration,
        "decisions": decisions,
        "ranking": ranking,
        "winner": winner,
        "control": CONTROL,
        "winner_passed": winner_passed,
        "selection_status": (
            f"Eligible: {winner}"
            if winner
            else "Tidak ada kandidat eligible pada selection 2024"
        ),
        "data_audit": _extended_data_audit(data),
        "state_audit": _state_audit(position_states),
        "classification_selection": classification["selection"],
        "classification_locked": classification["locked"],
        "classification_reference": classification["reference"],
        "v7_reference": _reference_table(v7_payload, "ranking"),
    }


def _adaptive_thresholds(opportunities: pd.DataFrame) -> dict[str, float | str]:
    calibration = opportunities.loc[
        DEVELOPMENT_START:pd.Timestamp("2023-12-31 23:59:59")
    ]
    return {
        "edge_position_max": float(calibration["position_from_edge"].quantile(0.60)),
        "edge_atr_max": float(calibration["distance_edge_atr"].quantile(0.65)),
        "range_width_min": float(calibration["range_width_atr"].quantile(0.10)),
        "range_width_max": float(calibration["range_width_atr"].quantile(0.90)),
        "width_change_abs_max": float(
            calibration["range_width_change"].abs().quantile(0.75)
        ),
        "midpoint_drift_abs_max": float(
            calibration["midpoint_drift_atr"].abs().quantile(0.75)
        ),
        "atr_percentile_max": float(calibration["atr_percentile"].quantile(0.85)),
        "rejection_min": float(calibration["rejection_body_atr"].quantile(0.45)),
        "rr_strong": float(calibration["reward_risk"].quantile(0.60)),
        "calibration": "2022-2023 saja",
    }


def _candidate_signal_map(opportunities, v7_control, hazard_control, best, thresholds):
    adaptive = (
        opportunities["position_from_edge"].le(thresholds["edge_position_max"])
        & opportunities["distance_edge_atr"].le(thresholds["edge_atr_max"])
        & opportunities["range_width_atr"].between(
            thresholds["range_width_min"], thresholds["range_width_max"]
        )
        & opportunities["range_width_change"].abs().le(
            thresholds["width_change_abs_max"]
        )
        & opportunities["midpoint_drift_atr"].abs().le(
            thresholds["midpoint_drift_abs_max"]
        )
        & opportunities["atr_percentile"].le(thresholds["atr_percentile_max"])
    )
    rejection = opportunities["rejection_body_atr"].ge(
        thresholds["rejection_min"]
    )
    hazard = pd.Series(opportunities.index.isin(hazard_control.index), index=opportunities.index)
    quality_score = adaptive.astype(int) + rejection.astype(int) + opportunities[
        "reward_risk"
    ].ge(thresholds["rr_strong"]).astype(int)
    masks = {
        ADAPTIVE_BOUNDARY: adaptive,
        REJECTION_CONFIRMATION: rejection,
        BREAKOUT_HAZARD: hazard,
        BOUNDARY_HAZARD: adaptive & hazard,
        FULL_V8: hazard & quality_score.ge(2),
    }
    output = {CONTROL: v7_control.copy()}
    rows = [_audit_row(CONTROL, v7_control, opportunities, len(opportunities))]
    for candidate, mask in masks.items():
        selected = opportunities.loc[mask.fillna(False)].copy()
        signals = _opportunities_to_signals(selected, best, candidate)
        signals = _enrich_from_opportunities(signals, selected)
        output[candidate] = signals
        rows.append(_audit_row(candidate, signals, selected, len(opportunities)))
    return output, pd.DataFrame(rows)


def _enrich_from_opportunities(signals, opportunities):
    if signals.empty:
        return signals.assign(
            range_low=np.nan,
            range_high=np.nan,
            range_mid=np.nan,
            range_width_atr=np.nan,
            direction="",
        )
    source = opportunities.loc[
        ~opportunities.index.duplicated(keep="first")
    ].reindex(signals.index)
    return _enrich_signals(signals, source)


def _audit_row(candidate, signals, source, total):
    unique = source.loc[~source.index.duplicated(keep="first")]
    selected = unique.reindex(signals.index).dropna(how="all") if not signals.empty else unique.iloc[0:0]
    return {
        "Kandidat": candidate,
        "Opportunity": total,
        "Sinyal lolos": len(signals),
        "Retensi opportunity (%)": len(signals) / max(total, 1) * 100,
        "BUY": int(selected.get("direction", pd.Series(dtype=str)).eq("BUY").sum()),
        "SELL": int(selected.get("direction", pd.Series(dtype=str)).eq("SELL").sum()),
        "Median RR": float(selected.get("reward_risk", pd.Series(dtype=float)).median()),
        "Median edge ATR": float(selected.get("distance_edge_atr", pd.Series(dtype=float)).median()),
    }


def _simulate_candidates(data, signal_map, states, atr_m15, model_1h, model_3h, start, end):
    return {
        candidate: _simulate_policy(
            data,
            signals.loc[start:end],
            states,
            atr_m15,
            ExitPolicy(candidate, break_even_r=1.0),
            model_1h["threshold"],
            model_3h["threshold"],
        )
        for candidate, signals in signal_map.items()
    }


def _result_table(results, signal_map):
    return pd.DataFrame(
        [
            {
                "Kandidat": candidate,
                "Sinyal tersedia": len(
                    signal_map[candidate].loc[DEVELOPMENT_START:DEVELOPMENT_END]
                ),
                **_metric_values(result),
            }
            for candidate, result in results.items()
        ]
    )


def _period_validation(results, signal_map):
    rows = []
    for label, start, end in _periods()[:-1]:
        for candidate, result in results.items():
            rows.append(
                {
                    "Periode": label,
                    "Kandidat": candidate,
                    "Sinyal tersedia": len(signal_map[candidate].loc[start:end]),
                    **_ledger_metric_values(_trades_in_period(result.trades, start, end)),
                }
            )
    return pd.DataFrame(rows)


def _decision_table(development, reference, periods, folds, monte_carlo, concentration):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    mc = monte_carlo.set_index("Kandidat")
    concentrated = concentration.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        primary = folds[
            folds["Kandidat"].eq(candidate)
            & folds["Kelompok"].eq("Primary validation")
        ]
        criteria = {
            "Growth positif": float(dev.loc[candidate, "Growth (%)"]) > 0,
            "PF development >= 1.30": float(dev.loc[candidate, "Profit factor"]) >= 1.30,
            "DD <= 10%": float(dev.loc[candidate, "Max drawdown (%)"]) <= 10,
            "2024 positif": float(period.loc[("Selection 2024", candidate), "Growth (%)"]) > 0,
            "2025 positif": float(period.loc[("Locked 2025", candidate), "Growth (%)"]) > 0,
            "2026H1 positif": float(ref.loc[candidate, "Growth (%)"]) > 0,
            "PF 2026H1 >= 1.10": float(ref.loc[candidate, "Profit factor"]) >= 1.10,
            "Primary fold >= 6/8": int(primary["Profitable"].sum()) >= 6,
            "Monte Carlo rugi <= 10%": float(
                mc.loc[candidate, "Probabilitas equity akhir < modal awal (%)"]
            ) <= 10,
            "Transaksi development >= 30": int(dev.loc[candidate, "Transaksi"]) >= 30,
            "Transaksi 2026H1 >= 5": int(ref.loc[candidate, "Transaksi"]) >= 5,
            "Konsentrasi 5 profit <= 40%": float(
                concentrated.loc[candidate, "Konsentrasi 5 profit terbesar (%)"]
            ) <= 40,
        }
        rows.append(
            {
                "Kandidat": candidate,
                **criteria,
                "Primary fold profitable": int(primary["Profitable"].sum()),
                "Kriteria lolos": int(sum(criteria.values())),
                "Total kriteria": len(criteria),
                "Lulus": bool(all(criteria.values())),
            }
        )
    return pd.DataFrame(rows)


def _selection_ranking(development, reference, periods, decisions, filter_audit):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    decision = decisions.set_index("Kandidat")
    audit = filter_audit.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        selection_growth = float(period.loc[("Selection 2024", candidate), "Growth (%)"])
        selection_pf = float(period.loc[("Selection 2024", candidate), "Profit factor"])
        selection_dd = float(period.loc[("Selection 2024", candidate), "Max drawdown (%)"])
        selection_trades = int(period.loc[("Selection 2024", candidate), "Transaksi"])
        safe_pf = 0.0 if not np.isfinite(selection_pf) else selection_pf
        score = selection_growth + min(safe_pf, 3.0) * 5 - selection_dd * 1.5 + min(selection_trades, 40) * 0.08
        rows.append(
            {
                "Kandidat": candidate,
                "Selection score 2024": score,
                "Growth selection 2024 (%)": selection_growth,
                "PF selection 2024": selection_pf,
                "DD selection 2024 (%)": selection_dd,
                "Transaksi selection 2024": selection_trades,
                "Selection eligible": selection_growth > 0 and selection_trades >= 8,
                "Growth development (%)": float(dev.loc[candidate, "Growth (%)"]),
                "PF development": float(dev.loc[candidate, "Profit factor"]),
                "DD development (%)": float(dev.loc[candidate, "Max drawdown (%)"]),
                "Transaksi development": int(dev.loc[candidate, "Transaksi"]),
                "Growth locked 2025 (%)": float(period.loc[("Locked 2025", candidate), "Growth (%)"]),
                "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
                "PF 2026H1": float(ref.loc[candidate, "Profit factor"]),
                "DD 2026H1 (%)": float(ref.loc[candidate, "Max drawdown (%)"]),
                "Transaksi 2026H1": int(ref.loc[candidate, "Transaksi"]),
                "Retensi opportunity (%)": float(audit.loc[candidate, "Retensi opportunity (%)"]),
                "Kriteria lolos": int(decision.loc[candidate, "Kriteria lolos"]),
                "Lulus": bool(decision.loc[candidate, "Lulus"]),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["Selection score 2024", "DD selection 2024 (%)"],
        ascending=[False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _reference_table(payload, key):
    if not payload or not isinstance(payload.get(key), pd.DataFrame):
        return pd.DataFrame()
    return payload[key].copy()
