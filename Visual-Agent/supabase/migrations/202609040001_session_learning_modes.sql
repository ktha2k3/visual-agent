alter table public.tutoring_sessions
  add column if not exists learning_mode text not null default 'guided_problem';

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'tutoring_sessions_learning_mode_check'
      and conrelid = 'public.tutoring_sessions'::regclass
  ) then
    alter table public.tutoring_sessions
      add constraint tutoring_sessions_learning_mode_check
      check (learning_mode in ('concept', 'guided_problem'));
  end if;
end
$$;

comment on column public.tutoring_sessions.learning_mode is
  'Persistent tutor policy: concept or guided_problem.';
