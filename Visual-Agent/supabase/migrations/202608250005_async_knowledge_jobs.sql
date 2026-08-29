begin;

alter table public.content_documents
  drop constraint if exists content_documents_size_bytes_check;
alter table public.content_documents
  add constraint content_documents_size_bytes_check
  check (size_bytes is null or size_bytes between 1 and 26214400);

create table if not exists public.knowledge_ingestion_jobs (
  id uuid primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null check (length(title) between 3 and 300),
  subject text not null check (subject in ('math', 'physics', 'chemistry', 'biology')),
  original_filename text not null,
  storage_path text not null unique,
  rights_note text not null,
  skill_ids text[] not null default '{}',
  size_bytes integer not null check (size_bytes between 1 and 26214400),
  status text not null default 'pending'
    check (status in ('pending', 'processing', 'completed', 'failed')),
  document_id uuid references public.content_documents(id) on delete set null,
  chunks integer check (chunks is null or chunks >= 0),
  embedding_status text,
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists knowledge_jobs_user_created_idx
  on public.knowledge_ingestion_jobs(user_id, created_at desc);
create index if not exists knowledge_jobs_pending_idx
  on public.knowledge_ingestion_jobs(status, created_at)
  where status in ('pending', 'processing');

alter table public.knowledge_ingestion_jobs enable row level security;

drop policy if exists knowledge_jobs_owner_read on public.knowledge_ingestion_jobs;
create policy knowledge_jobs_owner_read on public.knowledge_ingestion_jobs
  for select to authenticated
  using ((select auth.uid()) = user_id);

commit;
