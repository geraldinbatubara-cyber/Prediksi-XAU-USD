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
from gold_forecast.v1_sideways_defense import _regime_features
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
from gold_forecast.v1_sideways_specialist_v8 import (
    FULL_V8,
    _adaptive_thresholds,
    _candidate_signal_map as _v8_candidate_signal_map,
    _enrich_from_opportunities,
)
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


CONTROL = "Full Sideways v8 Control"
MODERATE_REGIME = "Moderate Regime"
MODERATE_ENTRY = "Moderate Entry"
RELAXED_CONFIRMATION = "Relaxed Confirmation"
WEIGHTED_EVIDENCE = "Weighted Evidence 2-of-3"
FULL_EXPANSION = "Opportunity Expansion Full"
CANDIDATES = (
    CONTROL,
    MODERATE_REGIME,
    MODERATE_ENTRY,
    RELAXED_CONFIRMATION,
    WEIGHTED_EVIDENCE,
    FULL_EXPANSION,
)

ATR_EXIT_POLICIES = (
    ExitPolicy("Control BE 1R", break_even_r=1.0),
    ExitPolicy("Trail 1.5R / 1.5 ATR", break_even_r=1.0, trailing_r=1.5),
    ExitPolicy("Trail 2R / 1.5 ATR", break_even_r=1.0, trailing_r=2.0),
    ExitPolicy(
        "Trail 70% TP / 1.5 ATR",
        break_even_r=1.0,
        trailing_tp_fraction=0.70,
    ),
    ExitPolicy(
        "Trail 2R / 2 ATR",
        break_even_r=1.0,
        trailing_r=2.0,
        trailing_atr_multiplier=2.0,
    ),
)


def run_v1_sideways_specialist_v9(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
    v8_payload: dict[str, object] | None = None,
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
    strict_range = _range_quality_frame(features, h1)
    expanded_range = _expanded_range_frame(strict_range)
    strict_opportunities = _augment_opportunities(
        data,
        strict_range,
        _mean_reversion_opportunities(data, strict_range, m15, spread_limit),
    )
    opportunities = _augment_opportunities(
        data,
        expanded_range,
        _mean_reversion_opportunities(data, expanded_range, m15, spread_limit),
    )
    setup_times = pd.DatetimeIndex(opportunities["setup_time"])
    opportunities["strict_regime"] = (
        strict_range["range_confirmed"].reindex(setup_times).fillna(False).to_numpy()
    )

    strict_bundle = _model_bundle(strict_opportunities, best, seed_base=100)
    v8_thresholds = _adaptive_thresholds(strict_opportunities)
    v8_map, _ = _v8_candidate_signal_map(
        strict_opportunities,
        strict_bundle["hazard_control"],
        strict_bundle["hazard_control"],
        best,
        v8_thresholds,
    )
    v8_control = v8_map[FULL_V8].copy()
    v8_control["strategy"] = CONTROL
    v8_control["Strategi"] = CONTROL

    bundle = _model_bundle(opportunities, best, seed_base=160)
    thresholds = _expansion_thresholds(opportunities, bundle)
    signal_map, filter_audit = _candidate_signal_map(
        opportunities,
        v8_control,
        best,
        bundle,
        thresholds,
    )
    union_signals = pd.concat(signal_map.values()).sort_index()
    union_signals = union_signals.loc[~union_signals.index.duplicated(keep="first")]
    position_states = _build_position_states(data, expanded_range, union_signals)
    model_1h = _train_state_model(
        position_states, "adverse_before_tp_1h", "1 jam", 141
    )
    model_3h = _train_state_model(
        position_states, "adverse_before_tp_3h", "3 jam", 151
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
    atr_development_results = _simulate_exit_policies(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END],
        signal_map[MODERATE_REGIME],
        position_states,
        atr_m15,
        model_1h,
        model_3h,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )
    atr_reference_results = _simulate_exit_policies(
        data.loc[REFERENCE_START:REFERENCE_END],
        signal_map[MODERATE_REGIME],
        position_states,
        atr_m15,
        model_1h,
        model_3h,
        REFERENCE_START,
        REFERENCE_END,
    )
    atr_trailing_comparison = _atr_trailing_comparison(
        atr_development_results,
        atr_reference_results,
    )
    development = _result_table(development_results, signal_map)
    reference = _result_table(reference_results, signal_map)
    periods = _period_validation(development_results, signal_map)
    folds = _fold_evaluation(development_results)
    monte_carlo = _monte_carlo_summary(development_results)
    concentration = _profit_concentration(development_results)
    marginal = _marginal_trade_audit(development_results)
    decisions = _decision_table(
        development,
        reference,
        periods,
        monte_carlo,
        marginal,
    )
    ranking = _selection_ranking(
        development,
        reference,
        periods,
        decisions,
        filter_audit,
        marginal,
    )
    eligible = ranking.loc[ranking["Selection eligible"]]
    winner = str(eligible.iloc[0]["Kandidat"]) if not eligible.empty else ""
    winner_passed = bool(
        winner and decisions.set_index("Kandidat").loc[winner, "Lulus"]
    )
    classification = _state_classification_tables(
        position_states, model_1h, model_3h
    )
    return {
        "methodology": {
            "Name": "v1 Sideways Specialist v9 - Opportunity Expansion Lab",
            "Control": "Full Sideways v8 direkonstruksi tanpa perubahan rule.",
            "Moderate regime": (
                "Minimal 2/5 bukti sideways, sentuhan kedua sisi, dan persistensi "
                "range minimal satu M15 selesai."
            ),
            "Evidence": (
                "Boundary, rejection, dan outcome probability dihitung terpisah; "
                "weighted candidate membutuhkan minimal 2 dari 3 bukti."
            ),
            "Risk veto": (
                "Hazard breakout di atas kuantil ekstrem 2022-2023 tetap memblokir entry."
            ),
            "Execution": (
                "Lot 0.01, satu posisi, spread/slippage broker, swap BUY, TP midpoint, "
                "SL boundary, time stop 12 jam, dan break-even 1R."
            ),
            "Selection": "2024 saja.",
            "Locked confirmation": "2025.",
            "Historical reference": "2026H1; tidak dipakai untuk tuning atau ranking.",
            "Expansion gate": (
                "Transaksi +30%, marginal PF >=1.10, total PF >=1.30, DD <=10%, "
                "2025 dan 2026H1 positif, serta Monte Carlo rugi <=10%."
            ),
            "Live trading lock": "BUY Specialist v4 dan ledger paper live tidak diubah.",
            "ATR trailing lab": (
                "Entry Moderate Regime dikunci. Control BE 1R dibandingkan dengan "
                "empat aturan trailing; 2026H1 hanya historical reference."
            ),
        },
        "expansion_thresholds": thresholds,
        "filter_audit": filter_audit,
        "marginal_trade_audit": marginal,
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
            f"Eligible: {winner}" if winner else "Tidak ada kandidat eligible pada selection 2024"
        ),
        "data_audit": _extended_data_audit(data),
        "state_audit": _state_audit(position_states),
        "classification_selection": classification["selection"],
        "classification_locked": classification["locked"],
        "classification_reference": classification["reference"],
        "v8_reference": _reference_table(v8_payload, "ranking"),
        "atr_trailing_comparison": atr_trailing_comparison,
        "atr_trailing_exit_counts": _atr_exit_counts(atr_development_results),
    }


def _expanded_range_frame(strict: pd.DataFrame) -> pd.DataFrame:
    frame = strict.copy()
    frame["strict_range_confirmed"] = frame["range_confirmed"].fillna(False)
    expanded_quality = (
        frame["sideways_votes"].ge(2)
        & frame["range_width_atr"].between(1.0, 7.0)
        & frame["touch_lower"].ge(1)
        & frame["touch_upper"].ge(1)
    )
    frame["range_quality"] = expanded_quality
    frame["range_confirmed"] = (
        expanded_quality.rolling(15, min_periods=15).sum().ge(12)
    )
    return frame


def _model_bundle(
    opportunities: pd.DataFrame,
    best: dict[str, object],
    *,
    seed_base: int,
):
    outcome, outcome_selection = _train_outcome_model(opportunities)
    persistence = _train_binary_model(
        opportunities, "persistence_12h", "Range Persistence", seed_base + 1
    )
    hazard = _train_binary_model(
        opportunities, "adverse_breakout_6h", "Adverse Breakout Hazard", seed_base + 11
    )
    directional, _ = _directional_thresholds(opportunities, outcome["probability"])
    sessions, _ = _session_selection(opportunities)
    signals, _ = _v2_candidate_signals(
        opportunities,
        outcome,
        persistence,
        hazard,
        directional,
        sessions,
        best,
    )
    hazard_control = _enrich_from_opportunities(
        signals["Breakout Hazard Gate"], opportunities
    )
    return {
        "outcome": outcome,
        "outcome_selection": outcome_selection,
        "persistence": persistence,
        "hazard": hazard,
        "hazard_control": hazard_control,
    }


def _expansion_thresholds(opportunities, bundle):
    calibration = opportunities.loc[
        DEVELOPMENT_START:pd.Timestamp("2023-12-31 23:59:59")
    ]
    hazard_probability = bundle["hazard"]["probability"].loc[calibration.index]
    outcome_probability = bundle["outcome"]["probability"].loc[calibration.index]
    adaptive = _adaptive_thresholds(opportunities)
    return {
        **adaptive,
        "edge_position_relaxed": float(calibration["position_from_edge"].quantile(0.75)),
        "edge_atr_relaxed": float(calibration["distance_edge_atr"].quantile(0.80)),
        "rejection_relaxed": float(calibration["rejection_body_atr"].quantile(0.30)),
        "outcome_moderate": float(outcome_probability.quantile(0.40)),
        "hazard_extreme": float(hazard_probability.quantile(0.80)),
        "calibration": "2022-2023 saja; 2024-2026 tidak menentukan threshold",
    }


def _candidate_signal_map(opportunities, control, best, bundle, thresholds):
    boundary = (
        opportunities["position_from_edge"].le(thresholds["edge_position_relaxed"])
        & opportunities["distance_edge_atr"].le(thresholds["edge_atr_relaxed"])
        & opportunities["range_width_atr"].between(1.0, 7.0)
    )
    rejection = opportunities["rejection_body_atr"].ge(
        thresholds["rejection_relaxed"]
    )
    outcome = bundle["outcome"]["probability"].ge(thresholds["outcome_moderate"])
    extreme_hazard = bundle["hazard"]["probability"].ge(thresholds["hazard_extreme"])
    strict = opportunities["strict_regime"].fillna(False)
    score = boundary.astype(int) + rejection.astype(int) + outcome.astype(int)
    masks = {
        MODERATE_REGIME: boundary & rejection & ~extreme_hazard,
        MODERATE_ENTRY: strict & (boundary | rejection) & outcome & ~extreme_hazard,
        RELAXED_CONFIRMATION: strict & boundary & (rejection | outcome) & ~extreme_hazard,
        WEIGHTED_EVIDENCE: score.ge(2) & ~extreme_hazard,
        FULL_EXPANSION: score.ge(1) & outcome & ~extreme_hazard,
    }
    output = {CONTROL: control.copy()}
    rows = [_audit_row(CONTROL, control, opportunities, len(opportunities))]
    for candidate, mask in masks.items():
        selected = opportunities.loc[mask.fillna(False)].copy()
        signals = _opportunities_to_signals(selected, best, candidate)
        signals = _enrich_from_opportunities(signals, selected)
        output[candidate] = signals
        rows.append(_audit_row(candidate, signals, selected, len(opportunities)))
    return output, pd.DataFrame(rows)


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


def _simulate_exit_policies(
    data,
    signals,
    states,
    atr_m15,
    model_1h,
    model_3h,
    start,
    end,
):
    return {
        policy.name: _simulate_policy(
            data,
            signals.loc[start:end],
            states,
            atr_m15,
            policy,
            model_1h["threshold"],
            model_3h["threshold"],
        )
        for policy in ATR_EXIT_POLICIES
    }


def _atr_trailing_comparison(development_results, reference_results):
    rows = []
    control_name = ATR_EXIT_POLICIES[0].name
    control_metrics = _metric_values(development_results[control_name])
    for policy in ATR_EXIT_POLICIES:
        development = _metric_values(development_results[policy.name])
        selection = _ledger_metric_values(
            _trades_in_period(
                development_results[policy.name].trades,
                SELECTION_START,
                SELECTION_END,
            )
        )
        locked = _ledger_metric_values(
            _trades_in_period(
                development_results[policy.name].trades,
                LOCKED_START,
                LOCKED_END,
            )
        )
        reference = _metric_values(reference_results[policy.name])
        trades = development_results[policy.name].trades
        average_net = float(trades["Net P/L"].mean()) if not trades.empty else np.nan
        rows.append(
            {
                "Kandidat": policy.name,
                "Aktivasi trailing": (
                    f"{policy.trailing_r:g}R"
                    if policy.trailing_r is not None
                    else (
                        f"{policy.trailing_tp_fraction:.0%} TP"
                        if policy.trailing_tp_fraction is not None
                        else "Tidak aktif"
                    )
                ),
                "Jarak trailing": (
                    f"{policy.trailing_atr_multiplier:g} ATR M15"
                    if policy.trailing_r is not None
                    or policy.trailing_tp_fraction is not None
                    else "-"
                ),
                "Transaksi development": int(development["Transaksi"]),
                "Growth development (%)": development["Growth (%)"],
                "PF development": development["Profit factor"],
                "DD development (%)": development["Max drawdown (%)"],
                "Rata-rata net/trade": average_net,
                "Growth selection 2024 (%)": selection["Growth (%)"],
                "Growth locked 2025 (%)": locked["Growth (%)"],
                "Growth 2026H1 (%)": reference["Growth (%)"],
                "PF 2026H1": reference["Profit factor"],
                "DD 2026H1 (%)": reference["Max drawdown (%)"],
                "Trailing aktif": int(trades["Trailing activated"].sum()),
                "BE aktif": int(trades["BE activated"].sum()),
                "Menjaga growth control": bool(
                    development["Growth (%)"] >= control_metrics["Growth (%)"] * 0.95
                ),
                "DD membaik": bool(
                    development["Max drawdown (%)"]
                    <= control_metrics["Max drawdown (%)"]
                ),
                "Locked positif": bool(locked["Growth (%)"] > 0),
                "Reference positif": bool(reference["Growth (%)"] > 0),
            }
        )
    frame = pd.DataFrame(rows)
    frame["Lulus vs control"] = (
        frame["Menjaga growth control"]
        & frame["DD membaik"]
        & frame["Locked positif"]
        & frame["Reference positif"]
    )
    return frame


def _atr_exit_counts(results):
    rows = []
    for candidate, result in results.items():
        counts = result.trades["Alasan exit"].value_counts()
        rows.append(
            {
                "Kandidat": candidate,
                "TP tersentuh": int(counts.get("TP tersentuh", 0)),
                "SL tersentuh": int(counts.get("SL tersentuh", 0)),
                "Break-even tersentuh": int(counts.get("Break-even tersentuh", 0)),
                "ATR trailing tersentuh": int(counts.get("ATR trailing tersentuh", 0)),
                "Time stop": int(counts.get("Time stop", 0)),
            }
        )
    return pd.DataFrame(rows)


def _result_table(results, signal_map):
    return pd.DataFrame(
        [
            {
                "Kandidat": candidate,
                "Sinyal tersedia": len(signal_map[candidate].loc[DEVELOPMENT_START:DEVELOPMENT_END]),
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


def _marginal_trade_audit(results):
    control = results[CONTROL].trades
    control_times = set(pd.to_datetime(control.get("Tanggal entry", pd.Series(dtype="datetime64[ns]"))))
    rows = []
    control_count = len(control)
    for candidate, result in results.items():
        trades = result.trades.copy()
        if candidate == CONTROL or trades.empty:
            additional = trades.iloc[0:0]
        else:
            additional = trades.loc[~pd.to_datetime(trades["Tanggal entry"]).isin(control_times)]
        metrics = _ledger_metric_values(additional)
        rows.append(
            {
                "Kandidat": candidate,
                "Transaksi total": len(trades),
                "Kenaikan transaksi (%)": (len(trades) / max(control_count, 1) - 1) * 100,
                "Transaksi tambahan": len(additional),
                "Marginal growth (%)": metrics["Growth (%)"],
                "Marginal PF": metrics["Profit factor"],
                "Marginal win rate (%)": metrics["Win rate (%)"],
            }
        )
    return pd.DataFrame(rows)


def _decision_table(development, reference, periods, monte_carlo, marginal):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    mc = monte_carlo.set_index("Kandidat")
    extra = marginal.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        marginal_pf = float(extra.loc[candidate, "Marginal PF"])
        criteria = {
            "Growth positif": float(dev.loc[candidate, "Growth (%)"]) > 0,
            "PF development >= 1.30": float(dev.loc[candidate, "Profit factor"]) >= 1.30,
            "DD <= 10%": float(dev.loc[candidate, "Max drawdown (%)"]) <= 10,
            "2024 positif": float(period.loc[("Selection 2024", candidate), "Growth (%)"]) > 0,
            "2025 positif": float(period.loc[("Locked 2025", candidate), "Growth (%)"]) > 0,
            "2026H1 positif": float(ref.loc[candidate, "Growth (%)"]) > 0,
            "Transaksi bertambah >= 30%": candidate == CONTROL or float(extra.loc[candidate, "Kenaikan transaksi (%)"]) >= 30,
            "Marginal PF >= 1.10": candidate == CONTROL or (np.isfinite(marginal_pf) and marginal_pf >= 1.10),
            "Monte Carlo rugi <= 10%": float(mc.loc[candidate, "Probabilitas equity akhir < modal awal (%)"]) <= 10,
            "Transaksi 2026H1 >= 5": int(ref.loc[candidate, "Transaksi"]) >= 5,
        }
        rows.append(
            {
                "Kandidat": candidate,
                **criteria,
                "Kriteria lolos": int(sum(criteria.values())),
                "Total kriteria": len(criteria),
                "Lulus": bool(all(criteria.values())),
            }
        )
    return pd.DataFrame(rows)


def _selection_ranking(development, reference, periods, decisions, filter_audit, marginal):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    decision = decisions.set_index("Kandidat")
    audit = filter_audit.set_index("Kandidat")
    extra = marginal.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        growth = float(period.loc[("Selection 2024", candidate), "Growth (%)"])
        pf = float(period.loc[("Selection 2024", candidate), "Profit factor"])
        dd = float(period.loc[("Selection 2024", candidate), "Max drawdown (%)"])
        trades = int(period.loc[("Selection 2024", candidate), "Transaksi"])
        safe_pf = 0.0 if not np.isfinite(pf) else pf
        score = growth + min(safe_pf, 3.0) * 5 - dd * 1.5 + min(trades, 40) * 0.08
        rows.append(
            {
                "Kandidat": candidate,
                "Selection score 2024": score,
                "Growth selection 2024 (%)": growth,
                "PF selection 2024": pf,
                "DD selection 2024 (%)": dd,
                "Transaksi selection 2024": trades,
                "Selection eligible": growth > 0 and trades >= 8,
                "Growth development (%)": float(dev.loc[candidate, "Growth (%)"]),
                "PF development": float(dev.loc[candidate, "Profit factor"]),
                "DD development (%)": float(dev.loc[candidate, "Max drawdown (%)"]),
                "Transaksi development": int(dev.loc[candidate, "Transaksi"]),
                "Growth locked 2025 (%)": float(period.loc[("Locked 2025", candidate), "Growth (%)"]),
                "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
                "PF 2026H1": float(ref.loc[candidate, "Profit factor"]),
                "Transaksi 2026H1": int(ref.loc[candidate, "Transaksi"]),
                "Retensi opportunity (%)": float(audit.loc[candidate, "Retensi opportunity (%)"]),
                "Kenaikan transaksi (%)": float(extra.loc[candidate, "Kenaikan transaksi (%)"]),
                "Marginal PF": float(extra.loc[candidate, "Marginal PF"]),
                "Kriteria lolos": int(decision.loc[candidate, "Kriteria lolos"]),
                "Lulus": bool(decision.loc[candidate, "Lulus"]),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["Selection score 2024", "DD selection 2024 (%)"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _reference_table(payload, key):
    if not payload or not isinstance(payload.get(key), pd.DataFrame):
        return pd.DataFrame()
    return payload[key].copy()
