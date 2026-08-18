-- Pulihkan baseline_v1 position 7 dari snapshot CLOSED terakhir yang sah.
with closed_snapshot as (
    select payload
    from public.paper_ledger_events
    where strategy_id = 'baseline_v1'
      and position_id = 7
      and event_type = 'POSITION_SNAPSHOT'
      and payload->>'status' = 'CLOSED'
    order by created_at desc
    limit 1
)
update public.paper_live_positions as position
set
    status = 'CLOSED',
    payload = closed_snapshot.payload,
    updated_at = now()
from closed_snapshot
where position.strategy_id = 'baseline_v1'
  and position.position_id = 7;

-- CLOSED bersifat terminal dan tidak boleh kembali menjadi SIGNAL atau OPEN.
create or replace function public.protect_closed_paper_position()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    if old.status = 'CLOSED' and new.status is distinct from 'CLOSED' then
        raise exception
            'Paper position %.% sudah CLOSED dan tidak boleh berubah menjadi %',
            old.strategy_id,
            old.position_id,
            new.status;
    end if;
    return new;
end;
$$;

drop trigger if exists protect_closed_paper_position_trigger
    on public.paper_live_positions;
create trigger protect_closed_paper_position_trigger
before update of status, payload on public.paper_live_positions
for each row execute function public.protect_closed_paper_position();

comment on function public.protect_closed_paper_position() is
    'Mencegah regresi snapshot paper trading setelah posisi berstatus CLOSED.';
