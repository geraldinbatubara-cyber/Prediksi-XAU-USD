from __future__ import annotations

import argparse
from email.message import EmailMessage
import json
import os
import smtplib
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


STRATEGY_LABELS = {
    "baseline_v1": "Baseline v1",
    "fixed_delay_5m": "Fixed Delay 5m",
    "buy_specialist_v4": "BUY Specialist v4",
    "sideways_moderate_regime": "Moderate Regime",
}


def format_notification(row: dict) -> str:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    notification_type = str(row.get("notification_type") or "").upper()
    strategy_id = str(row.get("strategy_id") or "-")
    strategy = STRATEGY_LABELS.get(strategy_id, strategy_id.replace("_", " ").title())
    position_id = _display(row.get("position_id"))
    direction = str(payload.get("arah") or "-").upper()
    lot = _number(payload.get("lot"), 2)

    if notification_type == "ENTRY":
        return "\n".join(
            [
                "GOLD PREDICTOR - NEW ENTRY",
                "",
                f"Strategi: {strategy}",
                f"Posisi: #{position_id} {direction}",
                f"Entry: USD {_number(payload.get('entry_price'))}",
                f"Lot: {lot}",
                f"TP: USD {_target_price(payload, take_profit=True)}",
                f"CL/SL: USD {_target_price(payload, take_profit=False)}",
                f"Waktu: {_display(payload.get('entry_time_wit'))}",
                f"Alasan: {_display(payload.get('catatan'))}",
            ]
        )

    return "\n".join(
        [
            "GOLD PREDICTOR - POSITION CLOSED",
            "",
            f"Strategi: {strategy}",
            f"Posisi: #{position_id} {direction}",
            f"Exit: {_display(payload.get('exit_reason'))}",
            f"Entry: USD {_number(payload.get('entry_price'))}",
            f"Exit price: USD {_number(payload.get('exit_price'))}",
            f"Gross P/L: USD {_signed(payload.get('gross_pl'))}",
            f"Swap: USD {_signed(payload.get('swap'))}",
            f"Net P/L: USD {_signed(payload.get('net_pl'))}",
            f"Waktu: {_display(payload.get('exit_time_wit'))}",
        ]
    )


def run_dispatcher(*, once: bool = False, interval_seconds: int = 30) -> None:
    config = _load_config()
    while True:
        try:
            processed = dispatch_pending(config)
            if processed:
                print(f"Email notifications processed: {processed}", flush=True)
        except Exception as exc:
            print(f"WARNING email dispatcher: {exc}", flush=True)
        if once:
            return
        time.sleep(max(10, interval_seconds))


def dispatch_pending(config: dict[str, str]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    rows = _supabase_request(
        config,
        "paper_notification_outbox",
        query={
            "select": "notification_id,strategy_id,position_id,notification_type,payload,attempt_count,status",
            "status": "in.(PENDING,FAILED)",
            "next_attempt_at": f"lte.{now}",
            "order": "created_at.asc",
            "limit": "20",
        },
    )
    processed = 0
    for row in rows or []:
        notification_id = row.get("notification_id")
        if not _claim(config, notification_id, str(row.get("status"))):
            continue
        try:
            provider_id = _send_email(config, row, format_notification(row))
            _update_outbox(
                config,
                notification_id,
                {
                    "status": "SENT",
                    "provider_message_id": provider_id,
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                    "last_error": None,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as exc:
            attempts = int(row.get("attempt_count") or 0) + 1
            delay = min(3600, 60 * (2 ** min(attempts - 1, 6)))
            _update_outbox(
                config,
                notification_id,
                {
                    "status": "FAILED",
                    "attempt_count": attempts,
                    "next_attempt_at": (
                        datetime.now(timezone.utc) + timedelta(seconds=delay)
                    ).isoformat(),
                    "last_error": str(exc)[:500],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        processed += 1
    return processed


def _claim(config: dict[str, str], notification_id: object, status: str) -> bool:
    rows = _supabase_request(
        config,
        "paper_notification_outbox",
        method="PATCH",
        query={"notification_id": f"eq.{notification_id}", "status": f"eq.{status}"},
        payload={
            "status": "PROCESSING",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        prefer="return=representation",
    )
    return bool(rows)


def _update_outbox(config: dict[str, str], notification_id: object, payload: dict) -> None:
    _supabase_request(
        config,
        "paper_notification_outbox",
        method="PATCH",
        query={"notification_id": f"eq.{notification_id}"},
        payload=payload,
        prefer="return=minimal",
    )


def _send_email(config: dict[str, str], row: dict, message: str) -> str:
    notification_type = str(row.get("notification_type") or "UPDATE").upper()
    strategy_id = str(row.get("strategy_id") or "-")
    strategy = STRATEGY_LABELS.get(strategy_id, strategy_id.replace("_", " ").title())
    email = EmailMessage()
    email["From"] = config["sender"]
    email["To"] = config["recipient"]
    email["Subject"] = f"Gold Predictor | {notification_type} | {strategy}"
    email.set_content(message)

    try:
        with smtplib.SMTP(config["smtp_host"], int(config["smtp_port"]), timeout=20) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(config["sender"], config["app_password"])
            refused = smtp.send_message(email)
    except (smtplib.SMTPException, OSError) as exc:
        raise RuntimeError("Email SMTP tidak dapat dikirim.") from exc
    if refused:
        raise RuntimeError("Server SMTP menolak alamat penerima.")
    return str(email["Message-ID"] or f"smtp:{row.get('notification_id')}")


def _supabase_request(
    config: dict[str, str],
    table: str,
    *,
    method: str = "GET",
    query: dict[str, str] | None = None,
    payload: dict | list | None = None,
    prefer: str | None = None,
):
    url = f"{config['supabase_url'].rstrip('/')}/rest/v1/{table}"
    if query:
        url = f"{url}?{urlencode(query)}"
    headers = {
        "apikey": config["supabase_key"],
        "Authorization": f"Bearer {config['supabase_key']}",
    }
    if prefer:
        headers["Prefer"] = prefer
    return _json_request(url, method=method, headers=headers, payload=payload)


def _json_request(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    payload: dict | list | None = None,
):
    request_headers = {"Accept": "application/json", **headers}
    data = None
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
    request = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            content = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Layanan notifikasi tidak dapat dihubungi.") from exc
    return json.loads(content) if content else None


def _load_config() -> dict[str, str]:
    config = {
        "supabase_url": os.getenv("SUPABASE_URL", "").strip(),
        "supabase_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip(),
        "sender": os.getenv("EMAIL_SENDER", "").strip(),
        "recipient": os.getenv("EMAIL_RECIPIENT", "").strip(),
        "app_password": os.getenv("EMAIL_APP_PASSWORD", "").strip().replace(" ", ""),
        "smtp_host": os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com").strip(),
        "smtp_port": os.getenv("EMAIL_SMTP_PORT", "587").strip(),
    }
    missing = [
        key
        for key in ("supabase_url", "supabase_key", "sender", "recipient", "app_password")
        if not config[key]
    ]
    if missing:
        raise RuntimeError(f"Konfigurasi belum lengkap: {', '.join(missing)}")
    return config


def _target_price(payload: dict, *, take_profit: bool) -> str:
    try:
        entry = float(payload.get("entry_price"))
        lot = float(payload.get("lot"))
        amount = float(payload.get("tp_usd" if take_profit else "cl_usd"))
        direction = str(payload.get("arah") or "").upper()
        points = amount / (lot * 100)
        sign = 1 if (direction == "BUY") == take_profit else -1
        return _number(entry + sign * points)
    except (TypeError, ValueError, ZeroDivisionError):
        return "-"


def _number(value: object, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "-"


def _signed(value: object) -> str:
    try:
        return f"{float(value):+,.2f}"
    except (TypeError, ValueError):
        return "-"


def _display(value: object) -> str:
    text = str(value or "").strip()
    return text if text else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold Predictor email dispatcher")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    run_dispatcher(once=args.once, interval_seconds=args.interval)


if __name__ == "__main__":
    main()
