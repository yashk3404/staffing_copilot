# Changelog

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