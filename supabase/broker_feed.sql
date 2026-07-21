create table if not exists public.broker_latest_quote (
    symbol text primary key,
    timestamp_utc timestamptz not null,
    bid double precision not null,
    ask double precision not null,
    source text not null default 'MT5 Demo via Supabase',
    updated_at timestamptz not null default now(),
    constraint broker_latest_quote_valid check (ask >= bid)
);

create table if not exists public.broker_m1_bars (
    symbol text not null,
    timestamp_utc timestamptz not null,
    open double precision not null,
    high double precision not null,
    low double precision not null,
    close double precision not null,
    tick_volume bigint,
    spread_points integer,
    source text not null default 'MT5 Demo via Supabase',
    primary key (symbol, timestamp_utc),
    constraint broker_m1_bars_valid check (
        high >= greatest(open, close, low)
        and low <= least(open, close, high)
    )
);

create index if not exists broker_m1_bars_latest_idx
    on public.broker_m1_bars (symbol, timestamp_utc desc);

alter table public.broker_latest_quote enable row level security;
alter table public.broker_m1_bars enable row level security;

drop policy if exists "Public read latest broker quote" on public.broker_latest_quote;
create policy "Public read latest broker quote"
    on public.broker_latest_quote for select
    to anon, authenticated
    using (true);

drop policy if exists "Public read broker M1 bars" on public.broker_m1_bars;
create policy "Public read broker M1 bars"
    on public.broker_m1_bars for select
    to anon, authenticated
    using (true);

grant select on public.broker_latest_quote to anon, authenticated;
grant select on public.broker_m1_bars to anon, authenticated;
grant select, insert, update on public.broker_latest_quote to service_role;
grant select, insert, update on public.broker_m1_bars to service_role;
