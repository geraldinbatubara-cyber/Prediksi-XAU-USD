from __future__ import annotations

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import (
    OOS_END,
    OOS_START,
    _compact_curve,
    _entry_signals,
    _prepare_m1,
    _simulate_exact,
)


PROFIT_FACTOR_TARGET = 1.30
MAX_DRAWDOWN_PCT = 10.0
MONTE_CARLO_RUNS = 10_000
MONTE_CARLO_SEED = 20260722


def run_v1_robustness(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    oos_data = data.loc[(data.index >= OOS_START) & (data.index <= OOS_END)]
    _, leaderboard, daily_oos = frozen_payload["v1"]
    best = leaderboard.iloc[0].to_dict()
    signals = _entry_signals(data, signal_daily, best)

    scenario_rows = []
    baseline_result = None
    for spread_multiplier in (1.0, 1.5, 2.0):
        for slippage_points in (2.0, 4.0, 6.0):
            result = _simulate_exact(
                oos_data,
                signals,
                best,
                "v1",
                spread_multiplier=spread_multiplier,
                slippage_points=slippage_points,
            )
            if spread_multiplier == 1.0 and slippage_points == 2.0:
                baseline_result = result
            summary = result.summary
            drawdown_pct = summary["Max drawdown"] / summary["Modal awal"] * 100
            scenario_rows.append(
                {
                    "Spread multiplier": spread_multiplier,
                    "Slippage points/sisi": slippage_points,
                    "Equity akhir": summary["Equity akhir"],
                    "Growth (%)": summary["Growth total"],
                    "Max drawdown": summary["Max drawdown"],
                    "Max drawdown (%)": drawdown_pct,
                    "Transaksi": summary["Jumlah transaksi"],
                    "Win rate (%)": summary["Win rate"],
                    "Profit factor": summary["Profit factor"],
                    "Total spread": summary["Biaya spread"],
                    "Total slippage": summary["Biaya slippage"],
                    "Lulus minimum": bool(
                        summary["Growth total"] > 0
                        and summary["Profit factor"] >= PROFIT_FACTOR_TARGET
                        and drawdown_pct <= MAX_DRAWDOWN_PCT
                    ),
                }
            )
    if baseline_result is None:
        raise RuntimeError("Skenario baseline v1 tidak terbentuk.")

    scenarios = pd.DataFrame(scenario_rows)
    monthly = _monthly_summary(baseline_result)
    monte_carlo, monte_carlo_summary = _monte_carlo(baseline_result.trades)
    basis_audit = _basis_audit(signal_daily, data)
    baseline = baseline_result.summary
    baseline_drawdown_pct = baseline["Max drawdown"] / baseline["Modal awal"] * 100
    profitable_scenarios = int((scenarios["Growth (%)"] > 0).sum())
    passed_scenarios = int(scenarios["Lulus minimum"].sum())
    negative_months = int((monthly["Net P/L"] < 0).sum())
    status = (
        "LAYAK KANDIDAT PAPER TEST"
        if baseline["Growth total"] > 0
        and baseline["Profit factor"] >= PROFIT_FACTOR_TARGET
        and baseline_drawdown_pct <= MAX_DRAWDOWN_PCT
        and profitable_scenarios == len(scenarios)
        and monte_carlo_summary["Probabilitas equity akhir < modal awal (%)"] <= 10
        else "BELUM LAYAK REAL-MONEY - LANJUT PAPER TEST"
    )
    summary = {
        "Status": status,
        "Equity baseline": baseline["Equity akhir"],
        "Growth baseline (%)": baseline["Growth total"],
        "Profit factor baseline": baseline["Profit factor"],
        "Max drawdown baseline": baseline["Max drawdown"],
        "Max drawdown baseline (%)": baseline_drawdown_pct,
        "Skenario profitable": profitable_scenarios,
        "Total skenario": int(len(scenarios)),
        "Skenario lulus minimum": passed_scenarios,
        "Bulan negatif": negative_months,
        "Total bulan": int(len(monthly)),
        "Target profit factor": PROFIT_FACTOR_TARGET,
        "Batas max drawdown (%)": MAX_DRAWDOWN_PCT,
        **monte_carlo_summary,
    }
    return {
        "summary": summary,
        "scenarios": scenarios,
        "monthly": monthly,
        "monte_carlo": monte_carlo,
        "basis_audit": basis_audit,
        "baseline_result": _compact_curve(baseline_result),
        "leaderboard": pd.DataFrame([best]),
        "daily_oos": daily_oos,
    }


def _monthly_summary(result) -> pd.DataFrame:
    trades = result.trades.copy()
    trades["Tanggal tutup"] = pd.to_datetime(trades["Tanggal tutup"])
    trades["Bulan"] = trades["Tanggal tutup"].dt.to_period("M").astype(str)
    rows = []
    start_equity = 1000.0
    for month in pd.period_range("2026-01", "2026-06", freq="M"):
        month_key = str(month)
        subset = trades[trades["Bulan"] == month_key]
        net = pd.to_numeric(subset.get("Net P/L", pd.Series(dtype=float)), errors="coerce")
        net_total = float(net.sum()) if not net.empty else 0.0
        end_equity = start_equity + net_total
        rows.append(
            {
                "Bulan": month_key,
                "Equity awal": start_equity,
                "Net P/L": net_total,
                "Growth bulan (%)": net_total / start_equity * 100 if start_equity else np.nan,
                "Equity akhir": end_equity,
                "Transaksi": float(len(subset)),
                "Win rate (%)": float((net > 0).mean() * 100) if not net.empty else np.nan,
                "Profit factor": _profit_factor(net),
                "Spread": float(subset["Biaya spread"].sum()) if not subset.empty else 0.0,
                "Slippage": float(subset["Biaya slippage"].sum()) if not subset.empty else 0.0,
                "Swap": float(subset["Swap"].sum()) if not subset.empty else 0.0,
            }
        )
        start_equity = end_equity
    return pd.DataFrame(rows)


def _monte_carlo(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    pnl = pd.to_numeric(trades["Net P/L"], errors="coerce").dropna().to_numpy(dtype=float)
    if pnl.size == 0:
        return pd.DataFrame(), {}
    rng = np.random.default_rng(MONTE_CARLO_SEED)
    samples = rng.choice(pnl, size=(MONTE_CARLO_RUNS, pnl.size), replace=True)
    cumulative = 1000.0 + samples.cumsum(axis=1)
    ending = cumulative[:, -1]
    running_peak = np.maximum.accumulate(np.column_stack([np.full(MONTE_CARLO_RUNS, 1000.0), cumulative]), axis=1)
    paths = np.column_stack([np.full(MONTE_CARLO_RUNS, 1000.0), cumulative])
    drawdown = (running_peak - paths).max(axis=1)
    ruin = (paths <= 0).any(axis=1)
    frame = pd.DataFrame({"Equity akhir": ending, "Max drawdown": drawdown, "Ruin": ruin})
    summary = {
        "Monte Carlo runs": float(MONTE_CARLO_RUNS),
        "Monte Carlo equity P5": float(np.quantile(ending, 0.05)),
        "Monte Carlo equity median": float(np.median(ending)),
        "Monte Carlo equity P95": float(np.quantile(ending, 0.95)),
        "Monte Carlo drawdown P95": float(np.quantile(drawdown, 0.95)),
        "Probabilitas equity akhir < modal awal (%)": float((ending < 1000.0).mean() * 100),
        "Probabilitas ruin (%)": float(ruin.mean() * 100),
    }
    return frame.iloc[::20].reset_index(drop=True), summary


def _basis_audit(signal_daily: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    broker_daily = m1.resample("1D").agg({"Close": "last"}).dropna().rename(columns={"Close": "MT5 Close"})
    benchmark = signal_daily[["Close"]].rename(columns={"Close": "GC=F Close"})
    aligned = benchmark.join(broker_daily, how="inner").dropna()
    benchmark_return = aligned["GC=F Close"].pct_change()
    broker_return = aligned["MT5 Close"].pct_change()
    valid = pd.concat([benchmark_return, broker_return], axis=1).dropna()
    direction_match = np.sign(valid.iloc[:, 0]) == np.sign(valid.iloc[:, 1])
    basis = aligned["MT5 Close"] - aligned["GC=F Close"]
    basis_pct = basis / aligned["GC=F Close"] * 100
    return pd.DataFrame(
        [
            {"Pemeriksaan": "Hari harga sejajar", "Nilai": float(len(aligned)), "Interpretasi": "Jumlah observasi bersama"},
            {"Pemeriksaan": "Korelasi return harian", "Nilai": float(valid.iloc[:, 0].corr(valid.iloc[:, 1])), "Interpretasi": "Mendekati 1 lebih baik"},
            {"Pemeriksaan": "Kesesuaian arah harian (%)", "Nilai": float(direction_match.mean() * 100), "Interpretasi": "Persentase arah return sama"},
            {"Pemeriksaan": "Median absolute basis (USD)", "Nilai": float(basis.abs().median()), "Interpretasi": "Selisih level harga tipikal"},
            {"Pemeriksaan": "Median absolute basis (%)", "Nilai": float(basis_pct.abs().median()), "Interpretasi": "Risiko perbedaan instrumen"},
            {"Pemeriksaan": "P95 absolute basis (USD)", "Nilai": float(basis.abs().quantile(0.95)), "Interpretasi": "Selisih level pada kondisi ekstrem"},
        ]
    )


def _profit_factor(net: pd.Series) -> float:
    profit = float(net[net > 0].sum())
    loss = abs(float(net[net < 0].sum()))
    return np.nan if loss == 0 else profit / loss
