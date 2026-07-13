-- Item 25-adjacent fix.
--
-- save_candidate_pool()/get_candidate_pool() were deliberately
-- session-scoped (st.session_state) per 0001/0002's design -- see
-- project_store.py's old module docstring. In practice that meant a
-- custom project's match scores, "why not the runner-up" data, and
-- the Full Candidate Pool table all silently went blank the moment
-- st.session_state reset: a page refresh, a new browser session, or
-- (per item 22's known gap) just being logged back in later. This
-- migration is the "worth a deliberate follow-up" the docstring
-- flagged -- moves both into Postgres so they survive like everything
-- else already does.

-- One score per (project, role, employee) assignment -- was always
-- computed at solve time, just never stored past that request.
alter table assignments add column if not exists final_score numeric;

-- The full ranked candidate list per role, as JSON:
-- { "<role>": [ {employee_id, name, role, experience_years,
--                availability_pct, skills, final_score, eligible}, ... ] }
-- One column on projects rather than a new table -- it's read/written
-- as a whole per project (never queried by individual candidate), so
-- a normalized table would only add join overhead with no benefit.
alter table projects add column if not exists candidate_pool jsonb;