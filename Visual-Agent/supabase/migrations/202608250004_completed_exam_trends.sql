begin;

-- Never let a future or unfinished exam year influence recommendations. The
-- national exam is considered complete from July onward.
create or replace view public.exam_skill_trends
with (security_invoker = true) as
with cutoff as (
  select case
    when extract(month from current_date) >= 7 then extract(year from current_date)::integer
    else extract(year from current_date)::integer - 1
  end as latest_completed_year
), ranked_years as (
  select distinct q.exam_year,
         dense_rank() over (order by q.exam_year desc) as year_rank
  from public.exam_questions q
  cross join cutoff c
  where q.review_status = 'approved'
    and q.exam_year <= c.latest_completed_year
), expanded as (
  select q.subject,
         jsonb_array_elements_text(q.skill_ids) as skill_id,
         y.year_rank,
         case q.cognitive_level
           when 'recognition' then 1.0
           when 'understanding' then 2.0
           else 3.0
         end as weight
  from public.exam_questions q
  join ranked_years y using (exam_year)
  where q.review_status = 'approved' and y.year_rank <= 2
), totals as (
  select subject, skill_id, year_rank, sum(weight) as weight
  from expanded
  group by subject, skill_id, year_rank
), maxima as (
  select subject, year_rank, max(weight) as max_weight
  from totals
  group by subject, year_rank
)
select t.skill_id,
       t.subject,
       coalesce(max(t.weight / nullif(m.max_weight, 0)) filter (where t.year_rank = 1), 0)::real as latest_weight,
       coalesce(max(t.weight / nullif(m.max_weight, 0)) filter (where t.year_rank = 2), 0)::real as previous_weight
from totals t
join maxima m using (subject, year_rank)
group by t.skill_id, t.subject;

commit;
