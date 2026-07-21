from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from gold_forecast.broker_data import load_broker_bars, load_broker_quote


def _request_json(
    base_url: str,
    api_key: str,
    table: str,
    *,
    method: str = "GET",
    query: dict[str, str] | None = None,
    payload: list[dict[str, object]] | None = None,
    prefer: str | None = None,
) -> object:
    url = f"{base_url.rstrip('/')}/rest/v1/{table}"
    if query:
        url = f"{url}?{urlencode(query)}"
    headers = {
        "apikey": api_key,
        "Accept": "application/json",
    }
    if api_key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {api_key}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
    if prefer:
        headers["Prefer"] = prefer

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Supabase menolak request {table} ({exc.code}): {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Supabase tidak dapat dihubungi untuk tabel {table}.") from exc
    return json.loads(body) if body else None


def _clean_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(column): _clean_value(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def publish_broker_snapshot(
    base_url: str,
    service_role_key: str,
    bars: pd.DataFrame,
    quotes: pd.DataFrame,
) -> None:
    if quotes.empty:
        raise ValueError("Quote broker kosong; tidak ada data yang dikirim ke Supabase.")

    latest_quote = quotes.tail(1).copy()
    latest_quote["source"] = latest_quote["source"].astype(str) + " via Supabase"
    _request_json(
        base_url,
        service_role_key,
        "broker_latest_quote",
        method="POST",
        query={"on_conflict": "symbol"},
        payload=_records(latest_quote[["symbol", "timestamp_utc", "bid", "ask", "source"]]),
        prefer="resolution=merge-duplicates,return=minimal",
    )

    if bars.empty:
        return
    publish_bars = bars.copy()
    publish_bars["source"] = publish_bars["source"].astype(str) + " via Supabase"
    columns = [
        "symbol",
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread_points",
        "source",
    ]
    records = _records(publish_bars[columns])
    for start in range(0, len(records), 500):
        _request_json(
            base_url,
            service_role_key,
            "broker_m1_bars",
            method="POST",
            query={"on_conflict": "symbol,timestamp_utc"},
            payload=records[start : start + 500],
            prefer="resolution=merge-duplicates,return=minimal",
        )


def load_supabase_broker_feed(
    base_url: str,
    read_key: str,
    symbol: str = "XAUUSD",
    bars_limit: int = 3000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quote_rows = _request_json(
        base_url,
        read_key,
        "broker_latest_quote",
        query={
            "select": "timestamp_utc,bid,ask,symbol,source",
            "symbol": f"eq.{symbol}",
            "limit": "1",
        },
    )
    bar_rows = _request_json(
        base_url,
        read_key,
        "broker_m1_bars",
        query={
            "select": "timestamp_utc,open,high,low,close,tick_volume,spread_points,symbol,source",
            "symbol": f"eq.{symbol}",
            "order": "timestamp_utc.desc",
            "limit": str(max(1, min(bars_limit, 5000))),
        },
    )
    bars = load_broker_bars(pd.DataFrame(bar_rows or []))
    quotes = load_broker_quote(pd.DataFrame(quote_rows or []))
    return bars, quotes
