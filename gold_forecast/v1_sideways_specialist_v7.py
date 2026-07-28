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
from gold_forecast.v1_signal_quality import _completed_bars
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


CONTROL = "v6 Persistence 2xM15 Control"
MTF_REGIME = "MTF Regime Gate"
NATIVE_PERSISTENCE = "Native M15 Persistence"
RANGE_REENTRY = "Persistence + Range Re-entry"
REJECTION_ANTI_CHASE = "Persistence + Rejection + Anti-Chasing"
ADAPTIVE_FUSION = "MTF Adaptive Fusion"
CANDIDATES = (
    CONTROL,
    MTF_REGIME,
    NATIVE_PERSISTENCE,
    RANGE_REENTRY,
    REJECTION_ANTI_CHASE,
    ADAPTIVE_FUSION,
)
POLICY = ExitPolicy(CONTROL, break_even_r=1.0)


def run_v1_sideways_specialist_v7(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
    defense_payload: dict[str, object] | None = None,
    v6_payload: dict[str, object] | None = None,
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
    filter_frame, native_thresholds = _entry_filter_frame(
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
            "Name": "v1 Sideways Specialist v7 - MTF Regime, Native Persistence & Rejection Entry",
            "Frozen execution": (
                "Breakout Hazard Gate v2, lot 0.01, TP/SL, spread, slippage, "
                "swap, time stop 12 jam, satu posisi, dan M15 Break-Even 1R."
            ),
            "Defense classifier development": "2022-2023 saja",
            "Selection": "Trade entry 2024 saja",
            "Locked confirmation": "Trade entry 2025",
            "Historical reference": "Trade entry 2026H1; tidak memilih pemenang",
            "Selected H1 regime classifier": selected_name,
            "MTF architecture": (
                "H1 memberi izin rezim, dua candle M15 selesai mengonfirmasi "
                "persistence, dan M1 tetap menjadi waktu eksekusi."
            ),
            "Native M15 persistence": (
                "ADX, efficiency ratio, choppiness, trend strength, dan MA slope "
                "dihitung langsung dari candle M15 selesai; bukan forward-fill H1."
            ),
            "Entry refinement": (
                "Range re-entry mengharuskan harga kembali ke dalam range; "
                "rejection menilai wick candle M15; anti-chasing membatasi "
                "pergerakan setelah candle konfirmasi maksimal 0.50 ATR M15."
            ),
            "Live trading lock": "Strategi dan ledger Paper Live Trading tidak diubah.",
        },
        "selected_regime_config": selected_regime.__dict__,
        "native_m15_thresholds": native_thresholds,
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
        "v6_reference": _reference_table(v6_payload, "ranking"),
    }


def _native_m15_regime(data):
    bars = _completed_bars(data, "15min").copy()
    high, low, close = bars["High"], bars["Low"], bars["Close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=bars.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=bars.index,
    )
    true_range_sum = true_range.rolling(14).sum()
    plus_di = 100 * plus_dm.rolling(14).sum() / true_range_sum
    minus_di = 100 * minus_dm.rolling(14).sum() / true_range_sum
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    bars["adx"] = dx.rolling(14, min_periods=8).mean()
    bars["efficiency"] = close.diff(14).abs() / close.diff().abs().rolling(14).sum()
    range_span = high.rolling(14).max() - low.rolling(14).min()
    bars["choppiness"] = (
        100 * np.log10(true_range_sum / range_span) / np.log10(14)
    )
    bars["atr"] = atr
    bars["fast"] = close.ewm(span=10, adjust=False).mean()
    bars["slow"] = close.ewm(span=30, adjust=False).mean()
    bars["trend_strength"] = (bars["fast"] - bars["slow"]).abs() / atr
    bars["slope"] = bars["slow"].diff(3).abs() / atr

    development = bars.loc[
        DEVELOPMENT_START : pd.Timestamp("2023-12-31 23:59:59")
    ]
    thresholds = {
        "adx_max": float(development["adx"].quantile(0.35)),
        "efficiency_max": float(development["efficiency"].quantile(0.35)),
        "choppiness_min": float(development["choppiness"].quantile(0.65)),
        "trend_strength_max": float(
            development["trend_strength"].quantile(0.35)
        ),
        "slope_max": float(development["slope"].quantile(0.35)),
        "minimum_sideways_votes": 3,
        "calibration": "Candle M15 selesai pada 2022-2023 saja",
    }
    checks = pd.DataFrame(index=bars.index)
    checks["adx"] = bars["adx"] <= thresholds["adx_max"]
    checks["efficiency"] = (
        bars["efficiency"] <= thresholds["efficiency_max"]
    )
    checks["choppiness"] = (
        bars["choppiness"] >= thresholds["choppiness_min"]
    )
    checks["trend_strength"] = (
        bars["trend_strength"] <= thresholds["trend_strength_max"]
    )
    checks["slope"] = bars["slope"] <= thresholds["slope_max"]
    bars["sideways_votes"] = checks.fillna(False).sum(axis=1)
    bars["raw_sideways"] = (
        bars["sideways_votes"] >= thresholds["minimum_sideways_votes"]
    )
    bars["persistence_2x"] = (
        bars["raw_sideways"].rolling(2, min_periods=2).sum().eq(2)
    )
    return bars.replace([np.inf, -np.inf], np.nan), thresholds


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
    frame["h1_sideways_votes"] = votes.reindex(index)
    frame["h1_raw_sideways"] = (
        frame["h1_sideways_votes"] >= config.minimum_sideways_votes
    )
    raw_m15 = raw_sideways.resample("15min").last().fillna(False)
    persistence = raw_m15.rolling(2, min_periods=2).sum().eq(2)
    frame["v6_persistence"] = (
        persistence.reindex(index, method="ffill").fillna(False)
    )

    native, thresholds = _native_m15_regime(data)
    aligned = native.reindex(index, method="ffill")
    frame["m15_sideways_votes"] = aligned["sideways_votes"]
    frame["m15_raw_sideways"] = aligned["raw_sideways"].fillna(False)
    frame["native_persistence"] = aligned["persistence_2x"].fillna(False)
    frame["m15_atr"] = aligned["atr"]
    frame["m15_close"] = aligned["Close"]
    frame["m15_previous_close"] = native["Close"].shift(1).reindex(
        index, method="ffill"
    )
    width = (signals["range_high"] - signals["range_low"]).replace(0, np.nan)
    buy = signals["direction"].eq("BUY")
    inside_range = aligned["Close"].between(
        signals["range_low"], signals["range_high"]
    )
    buy_reentry = (
        (aligned["Low"] <= signals["range_low"] + 0.15 * width)
        & (aligned["Close"] > signals["range_low"])
        & (aligned["Close"] <= signals["range_low"] + 0.45 * width)
    )
    sell_reentry = (
        (aligned["High"] >= signals["range_high"] - 0.15 * width)
        & (aligned["Close"] < signals["range_high"])
        & (aligned["Close"] >= signals["range_high"] - 0.45 * width)
    )
    frame["range_reentry"] = inside_range & np.where(
        buy, buy_reentry, sell_reentry
    )

    body = (aligned["Close"] - aligned["Open"]).abs().clip(
        lower=aligned["atr"] * 0.05
    )
    lower_wick = aligned[["Open", "Close"]].min(axis=1) - aligned["Low"]
    upper_wick = aligned["High"] - aligned[["Open", "Close"]].max(axis=1)
    buy_rejection = (
        (aligned["Close"] > aligned["Open"]) & (lower_wick / body >= 1.0)
    )
    sell_rejection = (
        (aligned["Close"] < aligned["Open"]) & (upper_wick / body >= 1.0)
    )
    frame["rejection"] = np.where(buy, buy_rejection, sell_rejection)

    favorable_move = np.where(
        buy,
        aligned["Close"] - frame["m15_previous_close"],
        frame["m15_previous_close"] - aligned["Close"],
    )
    frame["chase_distance_atr"] = (
        pd.Series(favorable_move, index=index).clip(lower=0)
        / aligned["atr"].replace(0, np.nan)
    )
    frame["anti_chasing"] = frame["chase_distance_atr"] <= 0.50
    return frame.replace([np.inf, -np.inf], np.nan), thresholds


def _candidate_signals(signals, frame):
    masks = {
        CONTROL: frame["v6_persistence"],
        MTF_REGIME: frame["h1_raw_sideways"] & frame["m15_raw_sideways"],
        NATIVE_PERSISTENCE: frame["native_persistence"],
        RANGE_REENTRY: frame["native_persistence"] & frame["range_reentry"],
        REJECTION_ANTI_CHASE: (
            frame["native_persistence"]
            & frame["rejection"]
            & frame["anti_chasing"]
        ),
        ADAPTIVE_FUSION: (
            frame["h1_raw_sideways"]
            & frame["native_persistence"]
            & frame["range_reentry"]
            & frame["rejection"]
            & frame["anti_chasing"]
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
                "H1 sideways (%)": selected["h1_raw_sideways"].mean() * 100,
                "M15 sideways (%)": selected["m15_raw_sideways"].mean() * 100,
                "Native persistence (%)": selected["native_persistence"].mean() * 100,
                "Range re-entry (%)": selected["range_reentry"].mean() * 100,
                "Rejection (%)": selected["rejection"].mean() * 100,
                "Anti-chasing (%)": selected["anti_chasing"].mean() * 100,
                "Median chase (ATR)": selected["chase_distance_atr"].median(),
            }
        )
    return pd.DataFrame(rows)


def _decision_table(development, reference, periods, folds, monte_carlo, concentration):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    mc = monte_carlo.set_index("Kandidat")
    concentrated = concentration.set_index("Kandidat")
    control_locked_growth = float(
        period.loc[("Locked 2025", CONTROL), "Growth (%)"]
    )
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
            "Growth 2025 >= 90% control": float(
                period.loc[("Locked 2025", candidate), "Growth (%)"]
            )
            >= control_locked_growth * 0.90,
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
            "Retensi sinyal >= 70%": (
                int(dev.loc[candidate, "Sinyal tersedia"])
                / max(int(dev.loc[CONTROL, "Sinyal tersedia"]), 1)
                * 100
                >= 70
            ),
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
