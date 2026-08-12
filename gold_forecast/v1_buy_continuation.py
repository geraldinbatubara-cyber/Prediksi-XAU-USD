from __future__ import annotations

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import _prepare_m1
from gold_forecast.v1_directional_specialization import (
    _fold_evaluation,
    _ledger_metric_values,
    _long_track_signals,
    _market_regime_state,
    _monte_carlo_summary,
    _period_validation,
    _result_table,
    _select_model_horizons,
    _simulate_all,
    _trade_entry_times,
    _train_hierarchical_candidates,
    _v3_candidate_inputs_with_placeholder,
    _v3_delay_candidates,
)
from gold_forecast.v1_entry_outcome import _balanced_signals
from gold_forecast.v1_entry_quality_path import (
    CONFIRMATION_END,
    CONFIRMATION_START,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
)
from gold_forecast.v1_fixed_delay import _build_fixed_delay_signals
from gold_forecast.v1_regime_classifier import _classifier_frame, _ohlc_bars
from gold_forecast.v1_regime_classifier_v3 import (
    LOCKED_END,
    LOCKED_START,
    THRESHOLD_END,
    VALIDATION_END,
    VALIDATION_START,
)
from gold_forecast.v1_risk_control import (
    RiskControlConfig,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_signal_quality import SignalQualityConfig, _entry_features
from gold_forecast.v1_unified_benchmark import _unified_best


CONTROL = "BUY Specialist v4 Control"
PULLBACK = "v4.1 Pullback Continuation"
BREAKOUT = "v4.1 Breakout Continuation"
COMBINED = "BUY Specialist v4.1 Combined"
CANDIDATES = (CONTROL, PULLBACK, BREAKOUT, COMBINED)
COOLDOWN_HOURS = 12


def run_v1_buy_continuation_lab(
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
    config = RiskControlConfig(
        "BUY Specialist v4.1",
        "Bullish continuation experiment",
        max_total_positions=1,
        max_same_direction=1,
    )

    balanced_config = SignalQualityConfig(
        "Balanced Entry Frozen",
        "Trend engine",
        conviction_multiplier=1.05,
        require_h1_trend=True,
        wait_hours=2,
    )
    balanced = _balanced_signals(
        data,
        signal_daily,
        best,
        entry_features,
        balanced_config,
        spread_limit,
        DEVELOPMENT_START,
        CONFIRMATION_END,
    )
    balanced = balanced.loc[~balanced.index.duplicated(keep="first")]

    base = _classifier_frame(data).drop(columns=["label"], errors="ignore")
    v3_runs, v3_selection = _train_hierarchical_candidates(base, data)
    selected = _select_model_horizons(v3_runs, v3_selection)
    v3_inputs, _ = _v3_candidate_inputs_with_placeholder(
        balanced, entry_features, selected
    )
    v3_signals, _ = _v3_delay_candidates(
        data, v3_inputs, selected, entry_features, best, spread_limit
    )
    adaptive = v3_signals["Ensemble Adaptive Confirmation"]
    regime_state, regime_audit = _market_regime_state(data, base)
    control = _long_track_signals(
        data, adaptive, regime_state, best, config
    )["Adaptive + Bear/Sideways Defense"]
    control = _tag_signals(control, CONTROL)

    raw_pullback, raw_breakout, gate_audit = _continuation_candidates(
        data,
        selected["Hierarchical Ensemble"],
        regime_state,
        best,
    )
    delayed_pullback, pullback_events = _build_fixed_delay_signals(
        data, raw_pullback, best, 5, spread_limit
    )
    delayed_breakout, breakout_events = _build_fixed_delay_signals(
        data, raw_breakout, best, 5, spread_limit
    )
    delayed_pullback = _apply_risk_pause(
        data, _cooldown_filter(_tag_signals(delayed_pullback, PULLBACK)), best, config
    )
    delayed_breakout = _apply_risk_pause(
        data, _cooldown_filter(_tag_signals(delayed_breakout, BREAKOUT)), best, config
    )
    combined = pd.concat([control, delayed_pullback, delayed_breakout]).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="first")]
    combined = _apply_risk_pause(data, combined, best, config)
    combined = _tag_signals(combined, COMBINED)

    signals = {
        CONTROL: control,
        PULLBACK: delayed_pullback,
        BREAKOUT: delayed_breakout,
        COMBINED: combined,
    }
    development_data = data.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    reference_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    development_results = _simulate_all(
        development_data, signals, best, config, DEVELOPMENT_START, DEVELOPMENT_END
    )
    reference_results = _simulate_all(
        reference_data, signals, best, config, CONFIRMATION_START, CONFIRMATION_END
    )
    development = _result_table(
        development_results, signals, DEVELOPMENT_START, DEVELOPMENT_END
    )
    reference = _result_table(
        reference_results, signals, CONFIRMATION_START, CONFIRMATION_END
    )
    periods = _period_validation(development_results, signals)
    folds = _fold_evaluation(development_results)
    monte_carlo = _monte_carlo_summary(development_results)
    regime_economic = _regime_audit(
        development_results, signals, regime_state
    )
    decisions = _decision_table(
        development, periods, folds, monte_carlo, regime_economic
    )
    ranking = _ranking(development, reference, decisions)
    winner = str(ranking.iloc[0]["Kandidat"])
    return {
        "methodology": {
            "Name": "BUY Specialist v4.1 - Bullish Continuation",
            "Control": "BUY Specialist v4 tidak diubah",
            "Development": "2022-2025",
            "Locked confirmation": "2025",
            "Historical reference": "2026H1; tidak memilih parameter",
            "Classifier persistence": "Dua candle H1 berturut-turut lolos moderate BUY",
            "Pullback": (
                "M15 bullish, low menyentuh EMA12, close reclaim EMA12, momentum positif, "
                "jarak maksimum 0.50 ATR"
            ),
            "Breakout": (
                "Close M15 menembus high 12 candle sebelumnya, momentum positif, "
                "jarak maksimum 1.00 ATR"
            ),
            "Execution": (
                "Fixed Delay 5m, spread <= P90 development, lot 0.01, TP USD 25, "
                "SL USD 10, maksimum satu posisi, cooldown 12 jam, loss pause v4"
            ),
            "Isolation": (
                "Tidak membaca atau mengubah ledger paper live BUY Specialist v4"
            ),
        },
        "regime_definition_audit": regime_audit,
        "gate_audit": gate_audit,
        "delay_audit": _delay_audit(
            raw_pullback, raw_breakout, delayed_pullback, delayed_breakout,
            pullback_events, breakout_events,
        ),
        "development": development,
        "historical_reference": reference,
        "period_validation": periods,
        "folds": folds,
        "monte_carlo_summary": monte_carlo,
        "regime_economic_audit": regime_economic,
        "decisions": decisions,
        "ranking": ranking,
        "winner": winner,
        "stress_summary": pd.DataFrame(),
    }


def _continuation_candidates(data, ensemble, regime_state, best):
    probabilities = ensemble["probabilities"].copy()
    thresholds = ensemble["thresholds"]
    classifier_buy = (
        probabilities["trend"].ge(float(thresholds.moderate_trend))
        & probabilities["direction_confidence"].ge(
            float(thresholds.moderate_direction)
        )
        & probabilities["up"].ge(0.50)
    )
    persistent_buy = classifier_buy.rolling(2, min_periods=2).sum().eq(2)

    m15 = _ohlc_bars(data, "15min")
    close, low, high = m15["Close"], m15["Low"], m15["High"]
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=48, adjust=False).mean()
    momentum = close.pct_change(4) * 100
    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    stretch = (close - fast).abs() / atr
    bullish_regime = regime_state.reindex(m15.index, method="ffill").eq("BULLISH")
    classifier_gate = persistent_buy.reindex(m15.index, method="ffill").fillna(False)
    trend_gate = (close > fast) & (fast > slow) & momentum.gt(0)
    pullback = (
        classifier_gate
        & bullish_regime
        & trend_gate
        & low.le(fast)
        & close.gt(fast)
        & stretch.le(0.50)
    )
    breakout_level = high.shift(1).rolling(12, min_periods=12).max()
    breakout = (
        classifier_gate
        & bullish_regime
        & trend_gate
        & close.gt(breakout_level)
        & stretch.le(1.00)
    )
    pullback = pullback & ~pullback.shift(1, fill_value=False)
    breakout = breakout & ~breakout.shift(1, fill_value=False)
    expected = max(float(best["Threshold entry (%)"]) * 1.05, 0.16)
    raw_pullback = _signal_frame(data, m15.index[pullback], expected, best, PULLBACK)
    raw_breakout = _signal_frame(data, m15.index[breakout], expected, best, BREAKOUT)
    audit = pd.DataFrame([
        {"Gerbang": "Observasi M15", "Jumlah": int(len(m15))},
        {"Gerbang": "Classifier BUY persisten 2xH1", "Jumlah": int(classifier_gate.sum())},
        {"Gerbang": "Classifier + regime BULLISH", "Jumlah": int((classifier_gate & bullish_regime).sum())},
        {"Gerbang": "Pullback reclaim baru", "Jumlah": int(pullback.sum())},
        {"Gerbang": "Breakout baru", "Jumlah": int(breakout.sum())},
    ])
    return raw_pullback, raw_breakout, audit


def _signal_frame(data, timestamps, expected, best, strategy):
    rows = []
    lot = float(best.get("Lot", 0.01) or 0.01)
    for timestamp in pd.DatetimeIndex(timestamps):
        location = data.index.searchsorted(timestamp, side="left")
        if location >= len(data):
            continue
        entry_time = pd.Timestamp(data.index[location])
        reference = float(data.iloc[location]["Close"])
        rows.append({
            "entry_time": entry_time,
            "signal_date": timestamp,
            "prediction": reference * (1 + expected / 100),
            "expected_change_pct": expected,
            "lot": lot,
            "strategy": strategy,
        })
    if not rows:
        return pd.DataFrame(
            columns=["signal_date", "prediction", "expected_change_pct", "lot", "strategy"]
        )
    return pd.DataFrame(rows).set_index("entry_time").sort_index()


def _cooldown_filter(signals, hours=COOLDOWN_HOURS):
    if signals.empty:
        return signals.copy()
    selected = []
    next_allowed = pd.Timestamp.min
    for timestamp in signals.sort_index().index:
        current = pd.Timestamp(timestamp)
        if current >= next_allowed:
            selected.append(current)
            next_allowed = current + pd.Timedelta(hours=hours)
    return signals.loc[selected].copy()


def _apply_risk_pause(data, signals, best, config):
    if signals.empty:
        return signals.copy()
    preliminary = _simulate_risk_control(data, signals, best, config)
    from gold_forecast.v1_directional_specialization import _loss_pause_filter
    return _loss_pause_filter(signals, preliminary.trades)


def _tag_signals(signals, strategy):
    output = signals.copy()
    if not output.empty:
        output["strategy"] = strategy
    return output


def _regime_audit(results, signals, regime_state):
    rows = []
    for candidate in CANDIDATES:
        frame = signals[candidate].loc[DEVELOPMENT_START:DEVELOPMENT_END]
        states = regime_state.reindex(frame.index, method="ffill").fillna("TRANSITION")
        trades = results[candidate].trades
        entry_times = _trade_entry_times(trades)
        if regime_state.index.tz is None:
            entry_times = entry_times.tz_convert(None)
        trade_states = regime_state.reindex(entry_times, method="ffill").fillna("TRANSITION")
        for regime in ("BULLISH", "BEARISH", "SIDEWAYS", "TRANSITION"):
            selected_trades = trades.loc[trade_states.eq(regime).to_numpy()] if not trades.empty else trades
            rows.append({
                "Kandidat": candidate,
                "Regime": regime,
                "Sinyal": int(states.eq(regime).sum()),
                **_ledger_metric_values(selected_trades),
            })
    return pd.DataFrame(rows)


def _decision_table(development, periods, folds, monte_carlo, regime):
    dev = development.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    mc = monte_carlo.set_index("Kandidat")
    regime_index = regime.set_index(["Kandidat", "Regime"])
    rows = []
    for candidate in CANDIDATES:
        primary = folds[
            folds["Kandidat"].eq(candidate)
            & folds["Kelompok"].eq("Primary validation")
        ]
        criteria = {
            "Growth positif": float(dev.loc[candidate, "Growth (%)"]) > 0,
            "PF >= 1.50": float(dev.loc[candidate, "Profit factor"]) >= 1.50,
            "DD <= 10%": float(dev.loc[candidate, "Max drawdown (%)"]) <= 10,
            "Bearish DD <= 5%": float(regime_index.loc[(candidate, "BEARISH"), "Max drawdown (%)"]) <= 5,
            "Sideways growth >= -2%": float(regime_index.loc[(candidate, "SIDEWAYS"), "Growth (%)"]) >= -2,
            "2024 positif": float(period.loc[("Model selection 2024", candidate), "Growth (%)"]) > 0,
            "2025 positif": float(period.loc[("Locked confirmation 2025", candidate), "Growth (%)"]) > 0,
            "Primary fold >= 6/8": int(primary["Profitable"].sum()) >= 6,
            "Monte Carlo rugi <= 10%": float(mc.loc[candidate, "Probabilitas equity akhir < modal awal (%)"]) <= 10,
        }
        rows.append({
            "Kandidat": candidate,
            **criteria,
            "Primary fold profitable": int(primary["Profitable"].sum()),
            "Kriteria lolos": int(sum(criteria.values())),
            "Total kriteria": len(criteria),
            "Lulus": bool(all(criteria.values())),
        })
    return pd.DataFrame(rows)


def _ranking(development, reference, decisions):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    decision = decisions.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        rows.append({
            "Kandidat": candidate,
            "Growth development (%)": float(dev.loc[candidate, "Growth (%)"]),
            "PF development": float(dev.loc[candidate, "Profit factor"]),
            "DD development (%)": float(dev.loc[candidate, "Max drawdown (%)"]),
            "Transaksi development": int(dev.loc[candidate, "Transaksi"]),
            "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
            "PF 2026H1": float(ref.loc[candidate, "Profit factor"]),
            "DD 2026H1 (%)": float(ref.loc[candidate, "Max drawdown (%)"]),
            "Transaksi 2026H1": int(ref.loc[candidate, "Transaksi"]),
            "Kriteria lolos": int(decision.loc[candidate, "Kriteria lolos"]),
            "Lulus": bool(decision.loc[candidate, "Lulus"]),
        })
    ranking = pd.DataFrame(rows).sort_values(
        ["Lulus", "Kriteria lolos", "PF development", "Growth development (%)"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _delay_audit(raw_pullback, raw_breakout, pullback, breakout, pullback_events, breakout_events):
    rows = []
    for name, raw, accepted, events in (
        (PULLBACK, raw_pullback, pullback, pullback_events),
        (BREAKOUT, raw_breakout, breakout, breakout_events),
    ):
        rows.append({
            "Kandidat": name,
            "Sinyal mentah": len(raw),
            "Lolos delay/cooldown/pause": len(accepted),
            "Batal barrier": int(events.get("expired", pd.Series(dtype=bool)).sum()),
            "Batal spread": int((~events["spread_ok"]).sum()) if not events.empty else 0,
        })
    return pd.DataFrame(rows)
