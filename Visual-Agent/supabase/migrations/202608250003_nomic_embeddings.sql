begin;

-- nomic-embed-text returns 768-dimensional vectors. Existing vectors are
-- derived data, so clear them before changing the pgvector typmod; re-upload or
-- re-embed documents after this migration.
drop index if exists public.content_chunks_embedding_idx;
drop function if exists public.match_content_chunks(extensions.vector, text, integer);
drop function if exists public.match_user_content_chunks(extensions.vector, uuid, text, integer);

update public.content_chunks set embedding = null where embedding is not null;
update public.content_documents
set embedding_status = 'lexical_fallback'
where embedding_status = 'ready';

alter table public.content_chunks
  alter column embedding type extensions.vector(768)
  using null::extensions.vector(768);

create index content_chunks_embedding_idx
  on public.content_chunks using hnsw (embedding extensions.vector_cosine_ops);

create or replace function public.match_content_chunks(
  query_embedding extensions.vector(768),
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
    and c.embedding is not null
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

create or replace function public.match_user_content_chunks(
  query_embedding extensions.vector(768),
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
