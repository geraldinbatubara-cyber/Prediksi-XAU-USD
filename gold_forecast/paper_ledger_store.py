from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from gold_forecast.supabase_broker import _request_json


_CONFIG = {
    "base_url": "",
    "read_key": "",
    "write_key": "",
}
_STATUS = {
    "mode": "CSV fallback",
    "last_error": "",
    "last_sync_utc": "",
    "decision_mode": "Not checked",
    "decision_last_error": "",
    "decision_last_sync_utc": "",
}

_STRATEGY_BY_FILENAME = {
    "live_trading_optimizer.csv": "baseline_v1",
    "live_trading_fixed_delay_5m.csv": "fixed_delay_5m",
    "live_trading_buy_specialist_v4.csv": "buy_specialist_v4",
    "live_trading_sideways_moderate.csv": "sideways_moderate_regime",
    "live_trading_optimizer_v10.csv": "optimizer_v10_archived",
}
_RECOVERY_POSITION_PATHS = {
    "baseline_v1": Path(
        "data/recovery/baseline_v1_positions_recovered_20260724.csv"
    ),
}
_RECOVERY_MANUAL_PATHS = {
    "baseline_v1": Path(
        "data/recovery/baseline_v1_manual_exits_recovered_20260724.csv"
    ),
}


def configure_paper_ledger_store(
    base_url: str,
    read_key: str,
    write_key: str,
) -> None:
    _CONFIG.update(
        {
            "base_url": str(base_url or "").strip(),
            "read_key": str(read_key or "").strip(),
            "write_key": str(write_key or "").strip(),
        }
    )
    _STATUS["mode"] = (
        "Supabase persistent"
        if _CONFIG["base_url"] and _CONFIG["read_key"] and _CONFIG["write_key"]
        else "CSV fallback"
    )


def paper_ledger_store_status() -> dict[str, str]:
    return dict(_STATUS)


def strategy_id_for_path(path: Path) -> str:
    return _STRATEGY_BY_FILENAME.get(
        Path(path).name,
        Path(path).stem.lower().replace("-", "_"),
    )


def load_recovery_positions(strategy_id: str) -> pd.DataFrame:
    return _load_recovery(_RECOVERY_POSITION_PATHS.get(strategy_id))


def load_recovery_manual_exits(strategy_id: str) -> pd.DataFrame:
    return _load_recovery(_RECOVERY_MANUAL_PATHS.get(strategy_id))


def load_persistent_positions(strategy_id: str) -> pd.DataFrame:
    rows = _load_rows(
        "paper_live_positions",
        strategy_id,
        "position_id,payload,updated_at",
    )
    return _payload_frame(rows)


def load_historical_max_position_id(strategy_id: str) -> int:
    if not _read_ready():
        return 0
    try:
        rows = _request_json(
            _CONFIG["base_url"],
            _CONFIG["read_key"],
            "paper_ledger_events",
            query={
                "select": "position_id",
                "strategy_id": f"eq.{strategy_id}",
                "position_id": "not.is.null",
                "order": "position_id.desc",
                "limit": "1",
            },
        )
        _sync_ok()
        if isinstance(rows, list) and rows:
            return _integer(rows[0].get("position_id")) or 0
    except Exception as exc:
        _sync_error(exc)
    return 0


def load_persistent_manual_exits(strategy_id: str) -> pd.DataFrame:
    rows = _load_rows(
        "paper_manual_exits",
        strategy_id,
        "manual_exit_id,position_id,payload,updated_at",
    )
    return _payload_frame(rows)


def save_persistent_positions(strategy_id: str, frame: pd.DataFrame) -> bool:
    if not _write_ready() or frame.empty:
        return False
    records = []
    events = []
    now = pd.Timestamp.now(tz="UTC").isoformat()
    for row in frame.to_dict(orient="records"):
        payload = _clean_mapping(row)
        position_id = _integer(payload.get("position_id"))
        if position_id is None:
            continue
        record = {
            "strategy_id": strategy_id,
            "position_id": position_id,
            "status": str(payload.get("status") or ""),
            "payload": payload,
            "updated_at": now,
        }
        records.append(record)
        events.append(
            _event_record(
                strategy_id,
                "POSITION_SNAPSHOT",
                position_id,
                payload,
                now,
            )
        )
    if not records:
        return False
    return _write_records(
        "paper_live_positions",
        "strategy_id,position_id",
        records,
        events,
    )


def save_persistent_manual_exits(strategy_id: str, frame: pd.DataFrame) -> bool:
    if not _write_ready() or frame.empty:
        return False
    records = []
    events = []
    now = pd.Timestamp.now(tz="UTC").isoformat()
    for row in frame.to_dict(orient="records"):
        payload = _clean_mapping(row)
        manual_exit_id = _integer(payload.get("manual_exit_id"))
        position_id = _integer(payload.get("position_id"))
        if manual_exit_id is None or position_id is None:
            continue
        record = {
            "strategy_id": strategy_id,
            "manual_exit_id": manual_exit_id,
            "position_id": position_id,
            "payload": payload,
            "updated_at": now,
        }
        records.append(record)
        events.append(
            _event_record(
                strategy_id,
                "MANUAL_EXIT_SNAPSHOT",
                position_id,
                payload,
                now,
            )
        )
    if not records:
        return False
    return _write_records(
        "paper_manual_exits",
        "strategy_id,manual_exit_id",
        records,
        events,
    )


def save_persistent_decision_snapshot(
    strategy_id: str,
    snapshot: dict[str, object],
) -> bool:
    if not _write_ready() or not snapshot:
        _STATUS["decision_mode"] = "Unavailable"
        _STATUS["decision_last_error"] = "Supabase write key belum tersedia."
        return False
    payload = _clean_mapping(snapshot)
    evaluation_key = str(payload.get("evaluation_key") or "").strip()
    decision_code = str(payload.get("decision_code") or "UNKNOWN").strip()
    if not evaluation_key:
        _STATUS["decision_mode"] = "Invalid snapshot"
        _STATUS["decision_last_error"] = "evaluation_key kosong."
        return False

    now = pd.Timestamp.now(tz="UTC").isoformat()
    record = {
        "strategy_id": strategy_id,
        "evaluation_key": evaluation_key,
        "decision_code": decision_code,
        "payload": payload,
        "updated_at": now,
    }
    event_identity = {
        "strategy_id": strategy_id,
        "evaluation_key": evaluation_key,
        "decision_code": decision_code,
        "active_signal_date": payload.get("active_signal_date"),
        "active_signal_direction": payload.get("active_signal_direction"),
        "trigger_status": payload.get("trigger_status"),
    }
    canonical = json.dumps(event_identity, sort_keys=True, separators=(",", ":"))
    event = {
        "event_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "strategy_id": strategy_id,
        "evaluation_key": evaluation_key,
        "decision_code": decision_code,
        "payload": payload,
        "created_at": now,
    }
    try:
        _request_json(
            _CONFIG["base_url"],
            _CONFIG["write_key"],
            "paper_decision_snapshots",
            method="POST",
            query={"on_conflict": "strategy_id,evaluation_key"},
            payload=[record],
            prefer="resolution=merge-duplicates,return=minimal",
        )
        _request_json(
            _CONFIG["base_url"],
            _CONFIG["write_key"],
            "paper_decision_events",
            method="POST",
            query={"on_conflict": "event_hash"},
            payload=[event],
            prefer="resolution=ignore-duplicates,return=minimal",
        )
        _STATUS["decision_mode"] = "Supabase persistent"
        _STATUS["decision_last_error"] = ""
        _STATUS["decision_last_sync_utc"] = now
        return True
    except Exception as exc:
        # Decision audit is additive; a missing audit table must never downgrade
        # the already-working position ledger to CSV fallback.
        _STATUS["decision_mode"] = "Schema/update required"
        _STATUS["decision_last_error"] = str(exc)[:300]
        return False


def merge_ledger_frames(
    persistent: pd.DataFrame,
    local: pd.DataFrame,
    recovery: pd.DataFrame,
    id_column: str,
) -> pd.DataFrame:
    frames = []
    for priority, frame in enumerate((recovery, local, persistent)):
        if frame is None or frame.empty:
            continue
        selected = frame.copy()
        numeric_ids = pd.to_numeric(selected.get(id_column), errors="coerce")
        selected = selected.loc[numeric_ids.notna()].copy()
        selected[id_column] = numeric_ids.loc[selected.index].astype(int)
        selected["_source_priority"] = priority
        selected["_business_time"] = _business_timestamp(selected)
        selected["_closed_priority"] = (
            selected["status"].astype(str).str.upper().eq("CLOSED").astype(int)
            if "status" in selected.columns
            else 0
        )
        frames.append(selected)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.sort_values(
        [id_column, "_closed_priority", "_business_time", "_source_priority"],
        na_position="first",
    )
    merged = merged.drop_duplicates(id_column, keep="last")
    return (
        merged.drop(columns=["_source_priority", "_business_time", "_closed_priority"])
        .sort_values(id_column)
        .reset_index(drop=True)
    )


def _load_rows(table: str, strategy_id: str, select: str) -> list[dict]:
    if not _read_ready():
        return []
    try:
        rows = _request_json(
            _CONFIG["base_url"],
            _CONFIG["read_key"],
            table,
            query={
                "select": select,
                "strategy_id": f"eq.{strategy_id}",
                "order": "updated_at.asc",
            },
        )
        _sync_ok()
        return list(rows or [])
    except Exception as exc:
        _sync_error(exc)
        return []


def _write_records(
    table: str,
    conflict_columns: str,
    records: list[dict],
    events: list[dict],
) -> bool:
    try:
        _request_json(
            _CONFIG["base_url"],
            _CONFIG["write_key"],
            table,
            method="POST",
            query={"on_conflict": conflict_columns},
            payload=records,
            prefer="resolution=merge-duplicates,return=minimal",
        )
        if events:
            _request_json(
                _CONFIG["base_url"],
                _CONFIG["write_key"],
                "paper_ledger_events",
                method="POST",
                query={"on_conflict": "event_hash"},
                payload=events,
                prefer="resolution=ignore-duplicates,return=minimal",
            )
        _sync_ok()
        return True
    except Exception as exc:
        _sync_error(exc)
        return False


def _payload_frame(rows: list[dict]) -> pd.DataFrame:
    payloads = [
        row["payload"]
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("payload"), dict)
    ]
    return pd.DataFrame(payloads)


def _event_record(
    strategy_id: str,
    event_type: str,
    position_id: int,
    payload: dict,
    created_at: str,
) -> dict:
    canonical = json.dumps(
        {
            "strategy_id": strategy_id,
            "event_type": event_type,
            "position_id": position_id,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "event_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "strategy_id": strategy_id,
        "event_type": event_type,
        "position_id": position_id,
        "payload": payload,
        "created_at": created_at,
    }


def _clean_mapping(row: dict) -> dict:
    return {str(key): _clean_value(value) for key, value in row.items()}


def _clean_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return _clean_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_clean_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _integer(value) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_recovery(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _business_timestamp(frame: pd.DataFrame) -> pd.Series:
    for column in (
        "last_update_wit",
        "manual_exit_time_wit",
        "exit_time_wit",
        "entry_time_wit",
    ):
        if column not in frame:
            continue
        parsed = pd.to_datetime(
            frame[column].astype(str).str.replace(" WIT", "", regex=False),
            errors="coerce",
        )
        if parsed.notna().any():
            return parsed
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")


def _read_ready() -> bool:
    return bool(_CONFIG["base_url"] and _CONFIG["read_key"])


def _write_ready() -> bool:
    return bool(_CONFIG["base_url"] and _CONFIG["write_key"])


def _sync_ok() -> None:
    _STATUS["mode"] = (
        "Supabase persistent" if _write_ready() else "Supabase read-only"
    )
    _STATUS["last_error"] = ""
    _STATUS["last_sync_utc"] = pd.Timestamp.now(tz="UTC").isoformat()


def _sync_error(exc: Exception) -> None:
    _STATUS["mode"] = "CSV fallback"
    _STATUS["last_error"] = str(exc)[:300]
