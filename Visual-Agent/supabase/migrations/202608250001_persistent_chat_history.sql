begin;

create table if not exists public.chat_messages (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid not null references public.tutoring_sessions(id) on delete cascade,
  request_id uuid not null,
  role text not null check (role in ('user', 'assistant')),
  content text not null check (char_length(content) between 1 and 50000),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (user_id, session_id, request_id, role)
);

create index if not exists chat_messages_user_session_created_idx
  on public.chat_messages(user_id, session_id, created_at);

alter table public.chat_messages enable row level security;
drop policy if exists owner_all on public.chat_messages;
create policy owner_all on public.chat_messages
  for all to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);

commit;
