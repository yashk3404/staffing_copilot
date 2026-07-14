# Changelog

## [v3.0] - Accounts & Supabase persistence

Everything below closes v2.1's biggest deferred decision: every custom
employee/project lived in `st.session_state` only, so a page refresh
or a new session lost it all. v3.0 adds real accounts and moves that
data into Postgres, with Row-Level Security as the actual enforcement
boundary rather than an app-side check. See `v3_roadmap.md` for the
item-by-item plan this followed.

### Added

- **Schema + Row-Level Security** (`supabase/migrations/0001_schema_and_rls.sql`)
  — `employees`, `projects`, `assignments` tables, each with a
  nullable `owner_id`: `NULL` means shared demo data visible to
  everyone, `owner_id = auth.uid()` means private to that user, and
  RLS policies enforce this at the database level. Postgres `CHECK`
  constraints (`experience_years >= 0`, `availability_pct BETWEEN 0
  AND 100`, etc.) provide server-side input validation, since the
  anon key is public and Streamlit-side checks alone aren't a real
  boundary.
- **Login / signup** (`src/auth.py`) — Supabase Auth (GoTrue) handles
  password hashing and rate-limiting server-side. Login errors are
  deliberately generic ("Invalid login credentials") so failed
  attempts don't leak whether an account exists.
- **Store internals swapped to Supabase** — `employee_store.py` /
  `project_store.py` now read/write Postgres instead of
  `st.session_state`, with every function signature kept identical so
  `dashboard.py`, `matcher.py`, and `optimize_staffing.py` needed zero
  changes. This was the exact seam Phase 3 (v1.0) was built to leave
  open.
- **A real multi-user id-collision bug, found and fixed
  mid-migration**: the schema initially made `employee_id`/`project_id`
  the sole primary key, but RLS means each user only ever sees their
  *own* rows when computing "next id" — two different users would
  both land on `CE001`/`C001` and the second insert would hard-fail.
  `0002_scope_ids_to_owner.sql` replaces that with a surrogate `uuid`
  primary key plus a composite `(id, owner_id)` uniqueness constraint.
  `tests/test_multi_user_id_scoping.py` (initially an empty stub)
  proves two users can each land on `CE001` without colliding, both
  against the mocked test double and against a real Supabase project.
- **RLS boundary tested against a real project, not mocked**
  (`tests/test_rls_boundary.py`) — signs in as two separate real test
  users and confirms User B can't read/update/delete User A's rows
  even querying broadly by `owner_id`, a spoofed `owner_id` insert
  fails outright, a regular user can't insert a `NULL`-owner ("demo")
  row, and two users can independently use the same app-facing id.
  Skips itself cleanly if the required env vars aren't set, so it
  never breaks a normal `pytest tests/` run or CI — this is the one
  place a mock would have hidden the actual risk.
- **Matcher/optimizer wired to the per-user merged pool** — a
  logged-in user's own saved employees are real scored candidates in
  their own custom-project solves, and invisible to every other
  user's pool, merged pool, and solve/candidate-pool results.
  `tests/test_multi_user_id_scoping.py` automates this end to end
  through the real `Matcher`/optimizer against `tests/fake_supabase.py`.
- **Deployed on Streamlit Cloud** against a production Supabase
  project, with connection secrets in Streamlit Cloud's secrets
  manager (never hardcoded).
- **Session persistence ("remember me")** — on login, if checked, the
  session's refresh token is written to a browser cookie via
  `streamlit-cookies-controller`. On every fresh script run with no
  logged-in user yet, that cookie is silently exchanged for a new
  session, closing item 22's original known gap (a refresh logging
  the user out).
- **Password reset flow** — "Forgot your password?" always shows the
  same generic success message whether or not the email is
  registered (same account-enumeration protection as the login
  error), and any send failure is caught silently for the same
  reason. The email links back to the app with `?token_hash=...&type=recovery`;
  clicking it authenticates the user into a dedicated "Set a new
  password" form rather than straight into the dashboard.
- **Email confirmation on signup** — same `token_hash`-based link
  shape, `?type=signup`.
- **`_app_url()`** (`src/auth.py`) — reads `APP_URL` from secrets/env
  so both the signup-confirmation and password-reset redirect targets
  point at the real deployed URL instead of Supabase's
  dashboard-configured default (`localhost:3000`), which was
  unreachable once this became a hosted app.
- **Custom email templates** — both the Confirm Signup and Reset
  Password templates were changed from Supabase's default
  `{{ .ConfirmationURL }}` link (which this server-rendered app can't
  consume — see `src/auth.py`'s module docstring for why) to
  `{{ .SiteURL }}?token_hash={{ .TokenHash }}&type=...`, and given a
  simple styled HTML layout (single button, inline CSS) instead of
  Supabase's plain-text default.
- **"🎬 Demo Mode"** — replaces the old session-only "Simulate" mode.
  Same batch-solve-and-compare engine, now explicitly scoped to the
  built-in 80-employee / 30-project demo roster (`owner_id IS NULL`)
  so it's fully usable without an account, and doesn't interact with
  a logged-in user's own data.
- **"👥 My Employees" mode** — search/filter, a table of every
  employee you've saved, and a delete action (with a confirm popup)
  per employee, via the new `employee_store.delete_employee()`.
- **"End Project" now fully removes the project**
  (`project_store.delete_project()`) instead of only clearing its
  assignments, confirmed via a popup before and after.
- **Candidate preview on Create Project** — an expandable table of
  your own employees (optionally filtered by role) before submitting,
  so "who's actually available" is visible up front.
- **`scripts/migrate_demo_roster_to_user.py`** — one-time import of
  the 80-employee demo roster into a real account's own `employees`
  rows, signing in as that user so RLS is respected rather than
  bypassed with a service-role key.
- **Match scores and candidate pools now persist in Postgres**
  (`0003_persist_scores_and_candidate_pool.sql` — a `final_score`
  column on `assignments`, a `candidate_pool jsonb` column on
  `projects`) instead of `st.session_state`, so a custom project's
  runner-up detail and Full Candidate Pool table survive a refresh, a
  new session, or logging back in later — closing the gap the v2.1
  `project_store.py` docstring had explicitly flagged as deferred.
- **Test suite** grown from 116 to 156 tests, including the new
  `test_rls_boundary.py` (10, real-project-only),
  `test_multi_user_id_scoping.py` (5), and expanded coverage in
  `test_employee_store.py` / `test_project_store.py` for the Supabase
  rewrite and the new delete/own-pool/candidate-pool-persistence
  behavior.

### Changed

- **Custom projects no longer match against the shared demo roster.**
  `employee_store.load_own_employees()` returns only the logged-in
  user's own saved employees; Create Project's capacity check,
  matcher solve, and result display all switched from the merged pool
  to this restricted one. `get_capacity_summary()` no longer merges
  internally — callers pass the already-final pool they want counted.
- **"Full Candidate Pool" for custom projects** now renders the saved
  ranked candidate pool per role, filtered to that role's actual job
  title (previously showed all saved employees under every role, a
  side effect of `match_adhoc()` scoring everyone against the role's
  query text for the solver's benefit — display now filters on
  `role_df['role'] == role` in addition to `eligible == True`,
  matching Demo Mode's existing table).
- **Production dependencies trimmed** — `requirements.txt` no longer
  includes `jupyter`, `matplotlib`, `seaborn`, `shap`, `faker`,
  `faiss-cpu`, or `groq` (the SDK; the app calls Groq's API via plain
  `requests` instead). All moved to the new `requirements-dev.txt`,
  which layers on top of `requirements.txt` for local notebook/test
  work. None of these are imported by the deployed `dashboard.py`.
- **`.streamlit/config.toml`** — file watcher disabled
  (`fileWatcherType = "none"`) to avoid unnecessary reloads on
  Streamlit Cloud.

### Fixed

- **Duplicate-employee-id bug from the demo-roster migration** —
  `migrate_demo_roster_to_user.py` intentionally reuses the CSV's
  E0xx ids, which produced duplicate index entries in
  `all_employees_df`. `.loc[emp_id]` on a duplicated index silently
  returns a `DataFrame` instead of a `Series`, which is why Assigned
  Team / Role Details were rendering pandas' raw repr instead of
  actual values. `load_all_employees()` now dedupes the merged frame
  on `employee_id`, keeping the custom/owned row.
- **`matcher.py`** — `_get_embedding_matrix()` crashed
  (`np.vstack([])`) on a genuinely empty `employees_df`, and
  `match()`/`match_adhoc()` crashed separately building an empty
  result (a columnless frame has no `final_score` column to sort by).
  Reachable any time a role's candidate pool is empty — e.g. a fresh
  user's `own_employees_df` — not just in a test. `dashboard.py`'s
  call sites happened to already guard on `.empty`, so this wasn't
  user-facing yet, but the underlying functions are now robust
  regardless of caller.
- **`tests/fake_supabase.py`** — `_FakeTable`/`_FakeQuery` had no
  `.update()`, so any test touching `save_candidate_pool()` (which
  calls `.update()` on the `projects` table) errored out, and had
  been silently failing 6 existing tests.
- **A migration ordering bug in `0002_scope_ids_to_owner.sql` itself**
  — the first version tried to drop `projects_pkey` while
  `assignments_project_id_fkey` still depended on it. Corrected to
  drop the dependent foreign key first, then the primary keys, then
  rebuild everything.
- **SMTP delivery** — Supabase's custom SMTP fields (Sender
  email/Host) were initially swapped, and separately the SMTP
  Username/Password were the Brevo account login rather than the
  dedicated SMTP key Brevo issues for this purpose — both silently
  failed to send (a `535 "Authentication failed"` SMTP rejection,
  visible only in Supabase's Auth Logs, never surfaced to the app,
  since `_login_form()`'s reset-password path deliberately swallows
  send exceptions to avoid leaking which emails are registered).

### Known limitations / deliberately deferred

- The shared demo roster (CSV + precomputed embeddings) deliberately
  stays outside Postgres — Option B from `v3_roadmap.md`. Migrating
  it too (Option A) is a possible v4+ move, not required for v3.0.
- Free-tier Supabase projects auto-pause after 7 days without API
  traffic; no keep-alive cron job is set up yet, so first load after
  a long idle period may take a moment to resume.
- CAPTCHA on login/signup was scoped in the original roadmap but
  deliberately dropped after three separate implementation attempts
  hit the same structural issue (Streamlit's sandboxed iframe has no
  synchronous channel back to Python for a client-side token).
  Supabase's own built-in per-IP/per-email rate limiting is the
  bot-protection baseline for now — see `src/auth.py`'s module
  docstring for the full reasoning and the real fix if this is
  revisited.

## [v2.1] - Live explanations & Simulate mode for custom projects

### Added
- **Candidate pool persistence** (`project_store.py`) —
  `save_candidate_pool()` / `get_candidate_pool()`, wired into
  `staff_custom_project()` right after matching and before the solve
  runs. Makes runner-up/explanation data available for ad-hoc
  projects the same way it already exists for premade ones.
- **`ContextRetriever.retrieve_adhoc()`** (`retrieve_context.py`) —
  RAG retrieval for custom-project assignments, built from the
  in-memory `project` + candidate pool instead of the offline
  CSV/JSON files that only cover the 30 premade projects. Same
  output shape as `retrieve()`; `generate_explanation.build_prompt()`
  consumes it unmodified. Missing pool / missing role / missing
  assignment all resolve to a clean `{"error": ...}` instead of a
  crash.
- **Live explanations wired into the dashboard** — the "Role Details
  & Explanations" panel now calls `retrieve_adhoc()` for custom
  projects instead of showing a "not available" caption.
- **`exclude_ids` support in `StaffingOptimizer`** — drop specific
  candidate rows before solving, so a batch run can exclude anyone
  already assigned in an earlier project within the same run. A
  full-exclude of a role's only eligible candidates now returns a
  clean `INFEASIBLE`/empty result instead of crashing.
- **Simulate mode** — pick any combination of custom projects and
  solve them together in one batch (jointly or staggered
  one-at-a-time), to compare planning-jointly vs reactive staffing
  outcomes on the same average-match-score metric.
- **Test suite** grown from 95 to 116 tests: 7 new `exclude_ids`
  tests in `test_optimizer.py`, 6 new candidate-pool round-trip
  tests in `test_project_store.py`, and a new `test_retrieve_context.py`
  (11 tests) checking schema parity between `retrieve()` and
  `retrieve_adhoc()`.

### Fixed
- `optimize_staffing.py`'s `INFEASIBLE` and no-feasible-solution
  early-return paths returned a bare columnless `pd.DataFrame()`,
  inconsistent with the "zero variables true" path's explicit
  columns. Consolidated into a shared `_empty_result()` helper used
  by all three empty-return paths.
- Runner-up panel in `dashboard.py` formatted `row['final_score']`
  with `:.4f`, which is always `None` for custom projects (they
  never get a `staffing_plan.csv` row) — this never crashed before
  because custom projects hit an earlier `continue` skip. Fixed to
  read the score from `ctx['assigned']['score']` instead, which is
  correct and populated either way.

### Known limitations / deliberately deferred
- Premade projects' "Full Candidate Pool" table is unchanged and
  still out of scope for this phase — still reads only
  `score_matrix.csv`.
- Persistence is still `st.session_state`-only; same deferred
  decision as v2.0 (see below). **Closed in v3.0, see above.**

## [v2.0] - Custom project & employee intake

### Added
- **Custom employee intake** (`employee_store.py`) — add employees via
  manual form or CV upload (PDF/DOCX/TXT), with CE0xx ids that never
  collide with the real E0xx roster.
- **Resume/CV parsing** (`resume_parser.py`) — extracts name, role,
  experience, skills, department, and location via the existing
  Ollama→Groq LLM pipeline; validates extracted skills against
  `skills_taxonomy.csv` (soft validation, never silently drops or
  auto-corrects); suggests taxonomy-constrained skill/role matches
  for the review form (exact → substring → fuzzy).
- **Duplicate detection** (`find_possible_duplicates`) — exact and
  fuzzy (typo-tolerant) name matching against both the real roster
  and session-added custom employees, soft-warn only, never blocks.
- **Custom project intake** (`project_store.py`) — create ad-hoc
  projects via a dashboard form, with C0xx ids that never collide
  with the real P0xx set. Includes a capacity pre-check
  (`get_capacity_summary`) shown before committing to a solve.
- **Ad-hoc matching & solving** (`matcher.match_adhoc`,
  `match_all_roles_adhoc`, `optimize_staffing.staff_custom_project`) —
  same scoring/CP-SAT pipeline as premade projects, for a project
  that hasn't been saved to `projects_with_index.csv`.
  `get_busy_employee_ids()` excludes anyone already staffed anywhere
  this session (premade plan + every custom project) so a new
  custom project's candidate pool never double-books someone.
- **Merged-pool matching** — `Matcher.match()` / `match_adhoc()` now
  accept an optional `employees_df`. When given a merged pool
  (`employee_store.load_all_employees()`), any employee not in the
  precomputed embeddings gets embedded on the fly from a profile
  string built the same way `embed_employees.py` builds one for the
  premade roster. This is what makes a custom employee a real
  candidate — not just a stored record — for both premade and
  custom-project matching. Fully backward compatible: omitting
  `employees_df` reproduces the exact pre-merge behavior.
- **Dashboard v2** — unified sidebar mode switch (Browse Projects /
  Create Project / Add Employee), Assigned Team and Role Details
  views read through the merged employee pool, visual polish (card
  styling for Project Details and Assigned Team, color-coded
  priority/budget badges, collapsible candidate-pool expanders).
- **Test suite** grown from 31 to 95 tests: new `test_employee_store.py`,
  `test_project_store.py`, `test_resume_parser.py`, plus merged-pool
  coverage added to `test_matcher.py` / `test_optimizer.py`.
  `tests/conftest.py` adds a fake `st.session_state` so the storage
  layer can be tested headless.

### Fixed
- `requirements.txt` was missing `pdfplumber` and `python-docx` —
  a fresh install would have crashed on the first resume upload.

### Known limitations / deliberately deferred
- Persistence is still `st.session_state`-only (in-memory, per
  session) — a real store (SQLite/Postgres/etc.) is a deliberately
  deferred decision, not an oversight. Every call site already reads
  through `employee_store.py` / `project_store.py`, so swapping the
  internals later shouldn't require touching callers. **Closed in
  v3.0, see above.**
- Premade projects' "Full Candidate Pool" table still reads the
  offline `score_matrix.csv`, so a custom employee won't appear
  there even though they're now eligible for custom-project solves —
  making that live would mean replacing a static file read with a
  per-render matcher call, a bigger change than what shipped here.
- Role-aware skill validation against `skills_taxonomy.csv`'s
  `related_roles` column (flagging a selected skill that doesn't fit
  the selected role) is scoped but not built.

## [v1.0-internship] - 30-day internship deliverable

Initial end-to-end proof of concept: synthetic data generation,
sentence-embedding-based semantic matching, CP-SAT conflict-free
optimization across 30 projects / 80 employees, RAG-grounded LLM
explanations (Ollama, Groq fallback for hosted deployment), SHAP
feature-importance breakdown, and the first Streamlit dashboard.
80 employees / 30 projects, 6,800 scored role-slots, 31-test suite.
See `docs/project_summary.md` for the full write-up of this
milestone. Tagged `v1.0-internship`.