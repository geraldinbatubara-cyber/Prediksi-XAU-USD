create table if not exists public.paper_live_positions (
    strategy_id text not null,
    position_id bigint not null,
    status text not null,
    payload jsonb not null,
    updated_at timestamptz not null default now(),
    primary key (strategy_id, position_id)
);

create table if not exists public.paper_manual_exits (
    strategy_id text not null,
    manual_exit_id bigint not null,
    position_id bigint not null,
    payload jsonb not null,
    updated_at timestamptz not null default now(),
    primary key (strategy_id, manual_exit_id)
);

create table if not exists public.paper_ledger_events (
    event_hash text primary key,
    strategy_id text not null,
    event_type text not null,
    position_id bigint,
    payload jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists paper_live_positions_strategy_idx
    on public.paper_live_positions (strategy_id, updated_at desc);

create index if not exists paper_manual_exits_strategy_idx
    on public.paper_manual_exits (strategy_id, updated_at desc);

create index if not exists paper_ledger_events_strategy_idx
    on public.paper_ledger_events (strategy_id, created_at desc);

alter table public.paper_live_positions enable row level security;
alter table public.paper_manual_exits enable row level security;
alter table public.paper_ledger_events enable row level security;

drop policy if exists "Public read paper positions" on public.paper_live_positions;
create policy "Public read paper positions"
    on public.paper_live_positions for select
    to anon, authenticated
    using (true);

drop policy if exists "Public read paper manual exits" on public.paper_manual_exits;
create policy "Public read paper manual exits"
    on public.paper_manual_exits for select
    to anon, authenticated
    using (true);

drop policy if exists "Public read paper ledger events" on public.paper_ledger_events;
create policy "Public read paper ledger events"
    on public.paper_ledger_events for select
    to anon, authenticated
    using (true);

grant select on public.paper_live_positions to anon, authenticated;
grant select on public.paper_manual_exits to anon, authenticated;
grant select on public.paper_ledger_events to anon, authenticated;

grant select, insert, update on public.paper_live_positions to service_role;
grant select, insert, update on public.paper_manual_exits to service_role;
grant select, insert on public.paper_ledger_events to service_role;
