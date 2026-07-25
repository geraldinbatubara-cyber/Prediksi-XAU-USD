from __future__ import annotations

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import POINT_SIZE, SLIPPAGE_POINTS
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
from gold_forecast.v1_sideways_specialist import _profit_concentration
from gold_forecast.v1_sideways_specialist_v3 import (
    _fold_evaluation,
    _periods,
    _state_audit,
    _state_classification_tables,
)
from gold_forecast.v1_sideways_specialist_v5_extensions import (
    ExitPolicy,
    _prepare_context,
    _simulate_policy,
)
from gold_forecast.v1_trend_strength_stability import _extended_data_audit


CONTROL = "v5B M15 BE 1R Control"
HARD_GATE = "Hard Defense Gate"
SOFT_GATE = "Soft Defense Score"
PERSISTENCE_GATE = "Persistence 2xM15"
ROOM_GATE = "Room-to-Target Gate"
ADAPTIVE_FUSION = "Adaptive Defense-to-Opportunity"
CANDIDATES = (
    CONTROL,
    HARD_GATE,
    SOFT_GATE,
    PERSISTENCE_GATE,
    ROOM_GATE,
    ADAPTIVE_FUSION,
)
POLICY = ExitPolicy(CONTROL, break_even_r=1.0)


def run_v1_sideways_specialist_v6(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
    defense_payload: dict[str, object] | None = None,
    v5_extensions_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    (
        data,
        control_signals,
        state_frame,
        atr_m15,
        model_1h,
        model_3h,
    ) = _prepare_context(gold_m1, frozen_payload)
    features, _, _ = _regime_features(data)
    development_data = data.loc[DEVELOPMENT_START:pd.Timestamp("2023-12-31 23:59:59")]
    regime_candidates = _regime_candidates(features, development_data)
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
    filter_frame = _entry_filter_frame(
        data, features, states, selected_regime, control_signals
    )
    candidate_signals = _candidate_signals(control_signals, filter_frame)

    development_results = _simulate_candidates(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END],
        candidate_signals,
        state_frame,
        atr_m15,
        model_1h,
        model_3h,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )
    reference_results = _simulate_candidates(
        data.loc[REFERENCE_START:REFERENCE_END],
        candidate_signals,
        state_frame,
        atr_m15,
        model_1h,
        model_3h,
        REFERENCE_START,
        REFERENCE_END,
    )
    development = _result_table(development_results, candidate_signals)
    reference = _result_table(reference_results, candidate_signals)
    periods = _period_validation(development_results, candidate_signals)
    folds = _fold_evaluation(development_results)
    monte_carlo = _monte_carlo_summary(development_results)
    concentration = _profit_concentration(development_results)
    filter_audit = _filter_audit(candidate_signals, filter_frame)
    decisions = _decision_table(
        development,
        reference,
        periods,
        folds,
        monte_carlo,
        concentration,
    )
    ranking = _selection_ranking(
        development, reference, periods, decisions, filter_audit
    )
    eligible = ranking.loc[ranking["Selection eligible"]]
    winner = str(eligible.iloc[0]["Kandidat"]) if not eligible.empty else ""
    fusion_eligible = eligible.loc[~eligible["Kandidat"].eq(CONTROL)]
    best_fusion = (
        str(fusion_eligible.iloc[0]["Kandidat"])
        if not fusion_eligible.empty
        else ""
    )
    stress_candidates = tuple(
        dict.fromkeys(candidate for candidate in (winner, best_fusion) if candidate)
    )
    stress = pd.concat(
        [
            _stress_test(
                data.loc[DEVELOPMENT_START:DEVELOPMENT_END],
                candidate_signals[candidate].loc[
                    DEVELOPMENT_START:DEVELOPMENT_END
                ],
                state_frame,
                atr_m15,
                model_1h,
                model_3h,
                candidate,
            )
            for candidate in stress_candidates
        ],
        ignore_index=True,
    ) if stress_candidates else pd.DataFrame()
    stress_counts = (
        stress.assign(Profitable=stress["Growth (%)"] > 0)
        .groupby("Kandidat")["Profitable"]
        .sum()
        .to_dict()
        if not stress.empty
        else {}
    )
    decisions = decisions.copy()
    decisions["Stress profitable"] = np.nan
    for candidate, count in stress_counts.items():
        decisions.loc[
            decisions["Kandidat"].eq(candidate), "Stress profitable"
        ] = int(count)
    winner_passed = False
    if winner:
        decision = decisions.loc[decisions["Kandidat"].eq(winner)].iloc[0]
        winner_passed = bool(decision["Lulus"]) and stress_counts.get(winner, 0) >= 7
        ranking.loc[
            ranking["Kandidat"].eq(winner), "Lulus termasuk stress"
        ] = winner_passed
    best_fusion_passed = False
    if best_fusion:
        fusion_decision = decisions.loc[
            decisions["Kandidat"].eq(best_fusion)
        ].iloc[0]
        best_fusion_passed = bool(
            fusion_decision["Lulus"]
            and stress_counts.get(best_fusion, 0) >= 7
        )
        ranking.loc[
            ranking["Kandidat"].eq(best_fusion), "Lulus termasuk stress"
        ] = best_fusion_passed

    classification = _state_classification_tables(
        state_frame, model_1h, model_3h
    )
    return {
        "methodology": {
            "Name": "v1 Sideways Specialist v6 - Defense-to-Opportunity Fusion",
            "Frozen execution": (
                "Breakout Hazard Gate v2, lot 0.01, TP/SL, spread, slippage, "
                "swap, time stop 12 jam, satu posisi, dan M15 Break-Even 1R."
            ),
            "Defense classifier development": "2022-2023 saja",
            "Selection": "Trade entry 2024 saja",
            "Locked confirmation": "Trade entry 2025",
            "Historical reference": "Trade entry 2026H1; tidak memilih pemenang",
            "Selected defense classifier": selected_name,
            "Soft score": (
                "+3 raw sideways, +2 range stabil, +1 ATR terkendali, "
                "+1 momentum netral, -3 tekanan tren, -2 ekspansi ATR."
            ),
            "Adaptive fusion": (
                "Soft score >= 4, raw sideways bertahan 2 candle M15, "
                "dan room-to-target setelah biaya memadai."
            ),
            "Live trading lock": "Strategi dan ledger Paper Live Trading tidak diubah.",
        },
        "selected_regime_config": selected_regime.__dict__,
        "classifier_development": classifier_development,
        "filter_audit": filter_audit,
        "development": development,
        "period_validation": periods,
        "historical_reference": reference,
        "folds": folds,
        "monte_carlo_summary": monte_carlo,
        "profit_concentration": concentration,
        "stress_summary": stress,
        "decisions": decisions,
        "ranking": ranking,
        "winner": winner,
        "best_fusion": best_fusion,
        "control": CONTROL,
        "winner_passed": winner_passed,
        "best_fusion_passed": best_fusion_passed,
        "selection_status": (
            f"Eligible: {winner}"
            if winner
            else "Tidak ada kandidat eligible pada selection 2024"
        ),
        "data_audit": _extended_data_audit(data),
        "state_audit": _state_audit(state_frame),
        "classification_selection": classification["selection"],
        "classification_locked": classification["locked"],
        "classification_reference": classification["reference"],
        "defense_reference": _reference_table(
            defense_payload, "strategy_validation"
        ),
        "v5b_reference": _v5b_reference(v5_extensions_payload),
    }


def _entry_filter_frame(data, features, states, config, signals):
    index = signals.index
    frame = features.reindex(index).copy()
    frame["state"] = states.reindex(index).fillna("UNCERTAIN")
    checks = pd.DataFrame(index=features.index)
    checks["adx"] = features["adx"] <= config.adx_max
    checks["efficiency"] = features["efficiency"] <= config.efficiency_max
    checks["choppiness"] = features["choppiness"] >= config.choppiness_min
    checks["trend_strength"] = (
        features["trend_strength"] <= config.trend_strength_max
    )
    checks["slope"] = features["slope"] <= config.slope_max
    votes = checks.fillna(False).sum(axis=1)
    raw_sideways = votes >= config.minimum_sideways_votes
    frame["sideways_votes"] = votes.reindex(index)
    frame["raw_sideways"] = (
        frame["sideways_votes"] >= config.minimum_sideways_votes
    )

    atr = features["atr"]
    frame["atr_change_3h"] = (
        atr / atr.shift(180).replace(0, np.nan) - 1
    ).reindex(index)
    atr_rank = atr.rolling(60 * 24, min_periods=24 * 10).rank(pct=True)
    frame["atr_percentile"] = atr_rank.reindex(index)
    stable_range = frame["atr_change_3h"].abs() <= 0.10
    controlled_atr = frame["atr_percentile"] <= 0.85
    neutral_momentum = frame["efficiency"] <= config.efficiency_max
    trend_pressure = (
        (frame["trend_strength"] > config.trend_strength_max * 1.20)
        | (frame["slope"] > config.slope_max * 1.20)
    )
    atr_expansion = frame["atr_change_3h"] > 0.20
    frame["defense_score"] = (
        frame["raw_sideways"].astype(int) * 3
        + stable_range.fillna(False).astype(int) * 2
        + controlled_atr.fillna(False).astype(int)
        + neutral_momentum.fillna(False).astype(int)
        - trend_pressure.fillna(False).astype(int) * 3
        - atr_expansion.fillna(False).astype(int) * 2
    )

    raw_m15 = raw_sideways.resample("15min").last().fillna(False)
    persistence = raw_m15.rolling(2, min_periods=2).sum().eq(2)
    frame["persistence_2x_m15"] = persistence.reindex(index, method="ffill").fillna(False)

    close = data["Close"].reindex(index)
    room = np.where(
        signals["direction"].eq("BUY"),
        signals["range_mid"] - close,
        close - signals["range_mid"],
    )
    round_trip_cost = (
        data["SpreadPoints"].reindex(index) * POINT_SIZE
        + 2 * SLIPPAGE_POINTS * POINT_SIZE
    )
    frame["room_to_target"] = room
    frame["round_trip_cost"] = round_trip_cost
    frame["room_gate"] = (
        pd.Series(room, index=index)
        >= signals["tp_usd"].astype(float) + round_trip_cost
    ) & (
        signals["tp_usd"].astype(float)
        / signals["sl_usd"].astype(float).replace(0, np.nan)
        >= 1.0
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def _candidate_signals(signals, frame):
    masks = {
        CONTROL: pd.Series(True, index=signals.index),
        HARD_GATE: frame["state"].eq("SIDEWAYS"),
        SOFT_GATE: frame["defense_score"].ge(4),
        PERSISTENCE_GATE: frame["persistence_2x_m15"],
        ROOM_GATE: frame["room_gate"],
        ADAPTIVE_FUSION: (
            frame["defense_score"].ge(4)
            & frame["persistence_2x_m15"]
            & frame["room_gate"]
        ),
    }
    output = {}
    for candidate, mask in masks.items():
        selected = signals.loc[mask.fillna(False)].copy()
        selected["Strategi"] = candidate
        output[candidate] = selected
    return output


def _simulate_candidates(
    data,
    signal_map,
    states,
    atr_m15,
    model_1h,
    model_3h,
    start,
    end,
):
    results = {}
    for candidate, signals in signal_map.items():
        policy = ExitPolicy(candidate, break_even_r=1.0)
        results[candidate] = _simulate_policy(
            data,
            signals.loc[start:end],
            states,
            atr_m15,
            policy,
            model_1h["threshold"],
            model_3h["threshold"],
        )
    return results


def _result_table(results, signal_map):
    rows = []
    for candidate, result in results.items():
        rows.append(
            {
                "Kandidat": candidate,
                "Sinyal tersedia": len(
                    signal_map[candidate].loc[DEVELOPMENT_START:DEVELOPMENT_END]
                ),
                **_metric_values(result),
            }
        )
    return pd.DataFrame(rows)


def _period_validation(results, signal_map):
    rows = []
    for label, start, end in _periods()[:-1]:
        for candidate, result in results.items():
            rows.append(
                {
                    "Periode": label,
                    "Kandidat": candidate,
                    "Sinyal tersedia": len(signal_map[candidate].loc[start:end]),
                    **_ledger_metric_values(
                        _trades_in_period(result.trades, start, end)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _filter_audit(signal_map, frame):
    control_count = max(len(signal_map[CONTROL]), 1)
    rows = []
    for candidate, signals in signal_map.items():
        selected = frame.reindex(signals.index)
        rows.append(
            {
                "Kandidat": candidate,
                "Sinyal lolos": len(signals),
                "Retensi sinyal (%)": len(signals) / control_count * 100,
                "Median defense score": selected["defense_score"].median(),
                "Sideways state (%)": selected["state"].eq("SIDEWAYS").mean() * 100,
                "Room gate (%)": selected["room_gate"].mean() * 100,
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
            "PF development >= 1.30": float(
                dev.loc[candidate, "Profit factor"]
            ) >= 1.30,
            "DD <= 10%": float(dev.loc[candidate, "Max drawdown (%)"]) <= 10,
            "2024 positif": float(
                period.loc[("Selection 2024", candidate), "Growth (%)"]
            ) > 0,
            "2025 positif": float(
                period.loc[("Locked 2025", candidate), "Growth (%)"]
            ) > 0,
            "2026H1 positif": float(ref.loc[candidate, "Growth (%)"]) > 0,
            "PF 2026H1 >= 1.10": float(
                ref.loc[candidate, "Profit factor"]
            ) >= 1.10,
            "Primary fold >= 6/8": int(primary["Profitable"].sum()) >= 6,
            "Monte Carlo rugi <= 10%": float(
                mc.loc[candidate, "Probabilitas equity akhir < modal awal (%)"]
            ) <= 10,
            "Transaksi development >= 30": int(
                dev.loc[candidate, "Transaksi"]
            ) >= 30,
            "Transaksi 2026H1 >= 5": int(
                ref.loc[candidate, "Transaksi"]
            ) >= 5,
            "Konsentrasi 5 profit <= 40%": float(
                concentrated.loc[
                    candidate, "Konsentrasi 5 profit terbesar (%)"
                ]
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
        selection_growth = float(
            period.loc[("Selection 2024", candidate), "Growth (%)"]
        )
        selection_pf = float(
            period.loc[("Selection 2024", candidate), "Profit factor"]
        )
        selection_dd = float(
            period.loc[("Selection 2024", candidate), "Max drawdown (%)"]
        )
        selection_trades = int(
            period.loc[("Selection 2024", candidate), "Transaksi"]
        )
        safe_pf = 0.0 if not np.isfinite(selection_pf) else selection_pf
        score = (
            selection_growth
            + min(safe_pf, 3.0) * 5
            - selection_dd * 1.5
            + min(selection_trades, 40) * 0.08
        )
        rows.append(
            {
                "Kandidat": candidate,
                "Selection score 2024": score,
                "Growth selection 2024 (%)": selection_growth,
                "PF selection 2024": selection_pf,
                "DD selection 2024 (%)": selection_dd,
                "Transaksi selection 2024": selection_trades,
                "Selection eligible": selection_growth > 0
                and selection_trades >= 8,
                "Growth development (%)": float(dev.loc[candidate, "Growth (%)"]),
                "PF development": float(dev.loc[candidate, "Profit factor"]),
                "DD development (%)": float(
                    dev.loc[candidate, "Max drawdown (%)"]
                ),
                "Transaksi development": int(dev.loc[candidate, "Transaksi"]),
                "Growth locked 2025 (%)": float(
                    period.loc[("Locked 2025", candidate), "Growth (%)"]
                ),
                "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
                "PF 2026H1": float(ref.loc[candidate, "Profit factor"]),
                "DD 2026H1 (%)": float(ref.loc[candidate, "Max drawdown (%)"]),
                "Transaksi 2026H1": int(ref.loc[candidate, "Transaksi"]),
                "Retensi sinyal (%)": float(
                    audit.loc[candidate, "Retensi sinyal (%)"]
                ),
                "Kriteria lolos": int(
                    decision.loc[candidate, "Kriteria lolos"]
                ),
                "Lulus": bool(decision.loc[candidate, "Lulus"]),
                "Lulus termasuk stress": False,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["Selection score 2024", "DD selection 2024 (%)"],
        ascending=[False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _stress_test(
    data, signals, states, atr_m15, model_1h, model_3h, candidate
):
    rows = []
    for spread_multiplier in (1.0, 1.25, 1.50):
        for slippage in (2.0, 4.0, 6.0):
            result = _simulate_policy(
                data,
                signals,
                states,
                atr_m15,
                ExitPolicy(candidate, break_even_r=1.0),
                model_1h["threshold"],
                model_3h["threshold"],
                spread_multiplier=spread_multiplier,
                slippage_points=slippage,
            )
            rows.append(
                {
                    "Kandidat": candidate,
                    "Spread multiplier": spread_multiplier,
                    "Slippage points": slippage,
                    **_metric_values(result),
                }
            )
    return pd.DataFrame(rows)


def _reference_table(payload, key):
    if not payload or not isinstance(payload.get(key), pd.DataFrame):
        return pd.DataFrame()
    return payload[key].copy()


def _v5b_reference(payload):
    if not payload or "v5b" not in payload:
        return pd.DataFrame()
    ranking = payload["v5b"].get("ranking")
    return ranking.head(3).copy() if isinstance(ranking, pd.DataFrame) else pd.DataFrame()
