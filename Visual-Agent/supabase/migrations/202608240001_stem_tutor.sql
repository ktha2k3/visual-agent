begin;

create extension if not exists vector with schema extensions;

create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  grade smallint not null default 12 check (grade between 10 and 12),
  locale text not null default 'vi',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.learning_goals (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  target_date date,
  created_at timestamptz not null default now()
);

create table if not exists public.selected_subjects (
  user_id uuid not null references auth.users(id) on delete cascade,
  subject text not null check (subject in ('math','physics','chemistry','biology')),
  primary key (user_id, subject)
);

create table if not exists public.curriculum_nodes (
  id text primary key,
  parent_id text references public.curriculum_nodes(id),
  subject text not null check (subject in ('math','physics','chemistry','biology')),
  grade smallint not null default 12,
  node_type text not null check (node_type in ('domain','topic','skill')),
  title text not null,
  outcome text,
  review_status text not null default 'pending' check (review_status in ('pending','approved','rejected')),
  created_at timestamptz not null default now()
);

create table if not exists public.content_documents (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  subject text not null check (subject in ('math','physics','chemistry','biology')),
  source_url text not null,
  publisher text,
  rights_note text,
  review_status text not null default 'pending' check (review_status in ('pending','approved','rejected')),
  created_at timestamptz not null default now()
);

create table if not exists public.content_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.content_documents(id) on delete cascade,
  title text,
  subject text not null check (subject in ('math','physics','chemistry','biology')),
  skill_ids jsonb not null default '[]'::jsonb,
  content text not null,
  source_url text,
  embedding extensions.vector(1024),
  review_status text not null default 'pending' check (review_status in ('pending','approved','rejected')),
  created_at timestamptz not null default now()
);

create index if not exists content_chunks_embedding_idx
  on public.content_chunks using hnsw (embedding extensions.vector_cosine_ops);
create index if not exists content_chunks_subject_idx
  on public.content_chunks(subject, review_status);

create table if not exists public.citations (
  id uuid primary key default gen_random_uuid(),
  content_chunk_id uuid not null references public.content_chunks(id) on delete cascade,
  title text not null,
  url text not null,
  excerpt text
);

create table if not exists public.assets (
  id uuid primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  storage_path text not null unique,
  mime_type text not null,
  size_bytes integer not null check (size_bytes > 0 and size_bytes <= 10485760),
  preview_url text,
  created_at timestamptz not null default now()
);

create index if not exists assets_user_id_idx on public.assets(user_id);

create table if not exists public.tutoring_sessions (
  id uuid primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  subject text check (subject in ('math','physics','chemistry','biology')),
  hint_level smallint not null default 0 check (hint_level between 0 and 4),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.problems (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid not null references public.tutoring_sessions(id) on delete cascade,
  asset_id uuid references public.assets(id) on delete set null,
  subject text not null check (subject in ('math','physics','chemistry','biology','unknown')),
  problem_markdown text not null,
  extraction jsonb not null default '{}'::jsonb,
  confidence real not null default 0 check (confidence between 0 and 1),
  created_at timestamptz not null default now()
);

create table if not exists public.attempts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  problem_id uuid references public.problems(id) on delete cascade,
  session_id uuid not null references public.tutoring_sessions(id) on delete cascade,
  answer text,
  answer_correct boolean,
  evidence real check (evidence between 0 and 1),
  misconception text,
  created_at timestamptz not null default now()
);

create table if not exists public.hint_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid not null references public.tutoring_sessions(id) on delete cascade,
  problem_id uuid references public.problems(id) on delete cascade,
  hint_level smallint not null check (hint_level between 0 and 4),
  content text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.mastery_records (
  user_id uuid not null references auth.users(id) on delete cascade,
  skill_id text not null references public.curriculum_nodes(id),
  subject text not null check (subject in ('math','physics','chemistry','biology')),
  mastery real not null default 0.5 check (mastery between 0 and 1),
  attempts integer not null default 0,
  recent_errors integer not null default 0,
  updated_at timestamptz not null default now(),
  primary key (user_id, skill_id)
);

create table if not exists public.recommendations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  skill_id text not null references public.curriculum_nodes(id),
  score real not null check (score between 0 and 1),
  reason text not null,
  generated_at timestamptz not null default now(),
  unique (user_id, skill_id)
);

create table if not exists public.exam_sources (
  id uuid primary key default gen_random_uuid(),
  url text not null,
  publisher text not null,
  exam_year smallint not null,
  subject text not null check (subject in ('math','physics','chemistry','biology')),
  exam_code text,
  checksum text,
  attribution text,
  fetched_at timestamptz,
  created_at timestamptz not null default now(),
  unique (url, subject)
);

create table if not exists public.exam_artifacts (
  id uuid primary key default gen_random_uuid(),
  source_id uuid not null references public.exam_sources(id) on delete cascade,
  storage_path text not null,
  original_url text not null,
  mime_type text not null,
  checksum text not null,
  created_at timestamptz not null default now(),
  unique (source_id, checksum)
);

create table if not exists public.exam_questions (
  id uuid primary key default gen_random_uuid(),
  source_id uuid references public.exam_sources(id) on delete cascade,
  exam_year smallint not null,
  subject text not null check (subject in ('math','physics','chemistry','biology')),
  exam_code text,
  question_number text not null,
  stem_markdown text not null,
  choices jsonb not null default '[]'::jsonb,
  correct_answer text,
  skill_ids jsonb not null default '[]'::jsonb,
  cognitive_level text not null default 'understanding' check (cognitive_level in ('recognition','understanding','application')),
  difficulty_estimate smallint not null default 2 check (difficulty_estimate between 1 and 5),
  difficulty_basis text not null default 'expert_estimate',
  confidence real not null default 0 check (confidence between 0 and 1),
  review_status text not null default 'pending' check (review_status in ('pending','approved','rejected')),
  raw_payload jsonb not null default '{}'::jsonb,
  reviewer_id uuid references auth.users(id),
  reviewed_at timestamptz,
  created_at timestamptz not null default now(),
  unique (source_id, question_number)
);

create index if not exists exam_questions_review_idx
  on public.exam_questions(review_status, subject, exam_year);

create table if not exists public.review_queue (
  id uuid primary key default gen_random_uuid(),
  question_id uuid not null unique references public.exam_questions(id) on delete cascade,
  status text not null default 'pending' check (status in ('pending','approved','rejected')),
  reviewer_id uuid references auth.users(id),
  notes text,
  created_at timestamptz not null default now(),
  reviewed_at timestamptz
);

create or replace function public.enqueue_exam_question()
returns trigger language plpgsql security definer set search_path = '' as $$
begin
  insert into public.review_queue(question_id) values (new.id)
  on conflict (question_id) do nothing;
  return new;
end;
$$;

drop trigger if exists enqueue_exam_question_trigger on public.exam_questions;
create trigger enqueue_exam_question_trigger
after insert on public.exam_questions
for each row execute function public.enqueue_exam_question();

create or replace view public.exam_skill_trends
with (security_invoker = true) as
with ranked_years as (
  select distinct exam_year,
         dense_rank() over (order by exam_year desc) as year_rank
  from public.exam_questions
  where review_status = 'approved'
), expanded as (
  select q.subject,
         jsonb_array_elements_text(q.skill_ids) as skill_id,
         y.year_rank,
         case q.cognitive_level when 'recognition' then 1.0 when 'understanding' then 2.0 else 3.0 end as weight
  from public.exam_questions q
  join ranked_years y using (exam_year)
  where q.review_status = 'approved' and y.year_rank <= 2
), totals as (
  select subject, skill_id, year_rank, sum(weight) as weight
  from expanded group by subject, skill_id, year_rank
), maxima as (
  select subject, year_rank, max(weight) as max_weight
  from totals group by subject, year_rank
)
select t.skill_id,
       t.subject,
       coalesce(max(t.weight / nullif(m.max_weight, 0)) filter (where t.year_rank = 1), 0)::real as latest_weight,
       coalesce(max(t.weight / nullif(m.max_weight, 0)) filter (where t.year_rank = 2), 0)::real as previous_weight
from totals t
join maxima m using (subject, year_rank)
group by t.skill_id, t.subject;

create or replace function public.match_content_chunks(
  query_embedding extensions.vector(1024),
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

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('uploads', 'uploads', false, 10485760, array['image/jpeg','image/png','image/gif','image/webp','image/bmp'])
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('exam-artifacts', 'exam-artifacts', false, 26214400, array['image/jpeg','image/png','image/webp','application/pdf'])
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

-- User-owned tables use the same compact policy pattern.
do $$
declare table_name text;
begin
  foreach table_name in array array[
    'profiles','learning_goals','selected_subjects','assets','tutoring_sessions',
    'problems','attempts','hint_events','mastery_records','recommendations'
  ] loop
    execute format('alter table public.%I enable row level security', table_name);
    execute format('drop policy if exists owner_all on public.%I', table_name);
    execute format(
      'create policy owner_all on public.%I for all to authenticated using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id)',
      table_name
    );
  end loop;
end $$;

alter table public.curriculum_nodes enable row level security;
alter table public.content_documents enable row level security;
alter table public.content_chunks enable row level security;
alter table public.citations enable row level security;
alter table public.exam_sources enable row level security;
alter table public.exam_artifacts enable row level security;
alter table public.exam_questions enable row level security;
alter table public.review_queue enable row level security;

create policy approved_curriculum_read on public.curriculum_nodes for select to authenticated
  using (review_status = 'approved');
create policy approved_documents_read on public.content_documents for select to authenticated
  using (review_status = 'approved');
create policy approved_chunks_read on public.content_chunks for select to authenticated
  using (review_status = 'approved');
create policy citations_read on public.citations for select to authenticated using (true);
create policy exam_sources_read on public.exam_sources for select to authenticated using (true);
create policy approved_exam_questions_read on public.exam_questions for select to authenticated
  using (review_status = 'approved');

create policy storage_owner_read on storage.objects for select to authenticated
  using (bucket_id = 'uploads' and (storage.foldername(name))[1] = (select auth.uid())::text);

-- Seed the MVP taxonomy. Content/outcomes still require reviewer approval.
insert into public.curriculum_nodes(id, subject, grade, node_type, title, review_status) values
('math.functions','math',12,'skill','Hàm số và đồ thị','approved'),
('math.derivatives','math',12,'skill','Đạo hàm và ứng dụng','approved'),
('math.integrals','math',12,'skill','Nguyên hàm và tích phân','approved'),
('math.spatial_geometry','math',12,'skill','Hình học không gian','approved'),
('math.probability','math',12,'skill','Xác suất và thống kê','approved'),
('physics.thermal','physics',12,'skill','Vật lí nhiệt','approved'),
('physics.gases','physics',12,'skill','Khí lí tưởng','approved'),
('physics.electromagnetism','physics',12,'skill','Điện từ học','approved'),
('physics.ac','physics',12,'skill','Dòng điện xoay chiều','approved'),
('physics.nuclear','physics',12,'skill','Vật lí hạt nhân','approved'),
('chemistry.esters','chemistry',12,'skill','Ester và lipid','approved'),
('chemistry.carbohydrates','chemistry',12,'skill','Carbohydrate','approved'),
('chemistry.amines','chemistry',12,'skill','Hợp chất chứa nitrogen','approved'),
('chemistry.polymers','chemistry',12,'skill','Polymer','approved'),
('chemistry.electrochemistry','chemistry',12,'skill','Pin điện và điện phân','approved'),
('biology.genetics','biology',12,'skill','Di truyền học','approved'),
('biology.evolution','biology',12,'skill','Tiến hóa','approved'),
('biology.ecology','biology',12,'skill','Sinh thái học','approved'),
('biology.cells','biology',12,'skill','Sinh học tế bào','approved'),
('biology.biotechnology','biology',12,'skill','Công nghệ sinh học','approved')
on conflict (id) do nothing;

commit;
