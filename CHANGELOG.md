# Changelog

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
  decision as v2.0 (see below).

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
  internals later shouldn't require touching callers.
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