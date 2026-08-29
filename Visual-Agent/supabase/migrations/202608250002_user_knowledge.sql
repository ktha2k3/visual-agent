begin;

-- User uploads are private. A NULL owner is reserved for reviewer-approved
-- curriculum content that is shared with every authenticated learner.
alter table public.content_documents
  add column if not exists user_id uuid references auth.users(id) on delete cascade,
  add column if not exists original_filename text,
  add column if not exists mime_type text,
  add column if not exists size_bytes integer
    check (size_bytes is null or size_bytes between 1 and 10485760),
  add column if not exists embedding_status text not null default 'ready'
    check (embedding_status in ('ready', 'lexical_fallback'));

alter table public.content_chunks
  add column if not exists user_id uuid references auth.users(id) on delete cascade;

create index if not exists content_documents_user_id_idx
  on public.content_documents(user_id, created_at desc);
create index if not exists content_chunks_user_subject_idx
  on public.content_chunks(user_id, subject, review_status);

drop policy if exists approved_documents_read on public.content_documents;
drop policy if exists approved_chunks_read on public.content_chunks;
drop policy if exists knowledge_documents_owner_all on public.content_documents;
drop policy if exists knowledge_chunks_owner_all on public.content_chunks;
drop policy if exists approved_global_documents_read on public.content_documents;
drop policy if exists approved_global_chunks_read on public.content_chunks;

create policy knowledge_documents_owner_all on public.content_documents
  for all to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);
create policy knowledge_chunks_owner_all on public.content_chunks
  for all to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);
create policy approved_global_documents_read on public.content_documents
  for select to authenticated
  using (user_id is null and review_status = 'approved');
create policy approved_global_chunks_read on public.content_chunks
  for select to authenticated
  using (user_id is null and review_status = 'approved');

drop policy if exists citations_read on public.citations;
drop policy if exists citations_for_visible_knowledge on public.citations;
create policy citations_for_visible_knowledge on public.citations
  for select to authenticated
  using (
    exists (
      select 1
      from public.content_chunks c
      where c.id = content_chunk_id
        and (
          c.user_id = (select auth.uid())
          or (c.user_id is null and c.review_status = 'approved')
        )
    )
  );

create or replace function public.match_user_content_chunks(
  query_embedding extensions.vector(1024),
  match_user_id uuid,
  match_subject text,
  match_count integer default 5
)
returns table(id uuid, title text, content text, source_url text, similarity real)
language sql stable security invoker set search_path = '' as $$
  select c.id, c.title, c.content, c.source_url,
         (1 - (c.embedding <=> query_embedding))::real as similarity
  from public.content_chunks c
  where c.review_status = 'approved'
    and c.subject = match_subject
    and (c.user_id = match_user_id or c.user_id is null)
    and c.embedding is not null
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

commit;
