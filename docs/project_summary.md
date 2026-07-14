# Staffing Copilot — Project Summary

**Built by:** Kumar Yash
**Role:** Python AI Intern, Yash Technologies  
**Duration:** 30 days  
**Type:** End-to-end AI system (proof of concept)

> This document is the write-up of the original 30-day internship
> deliverable (`v1.0-internship` tag) and is kept as-is for that
> record. For what's been built since — custom employee/project
> intake, resume parsing, the merged matching pool, the expanded
> test suite, live explanations and Simulate mode for custom
> projects, and real accounts with Postgres persistence and
> Row-Level Security — see the "Post-Internship Extensions (v2.0)",
> "Post-v2.0 Extensions (v2.1)", and "Post-v2.1 Extensions (v3.0)"
> sections at the end, and [`../CHANGELOG.md`](../CHANGELOG.md) for
> the full history.

---

## What It Does

An AI-powered employee-to-project matching and staffing optimization
system that:

1. **Semantically matches** employees to project roles using sentence
   embeddings and cosine similarity
2. **Clusters** employees by skill profile to validate embedding quality
3. **Optimally assigns** employees across multiple projects simultaneously
   using constraint-based optimization — no double-booking, no conflicts
4. **Explains** every staffing decision in plain English using a locally
   running LLM (zero API cost)
5. **Shows** which numeric features drove each match score using SHAP
6. **Presents** everything through an interactive Streamlit dashboard

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384-dim) | Semantic matching |
| Vector store | FAISS (IndexFlatIP) | Fast similarity search |
| Clustering | KMeans (scikit-learn) | Skill profile grouping |
| Optimization | Google OR-Tools (CP-SAT solver) | Conflict-free assignment |
| RAG retrieval | Custom ContextRetriever | Grounds LLM explanations |
| LLM | Ollama (llama3.2, local) + Groq fallback (llama-3.1-8b-instant, cloud) | Plain-English explanations |
| Explainability | SHAP (LinearExplainer) | Feature-level score breakdown |
| Dashboard | Streamlit | Interactive front end |
| Data | Synthetic (Faker, Indian locale) | 80 employees, 30 projects |
| Testing | pytest | 31 unit + integration tests |

---

## Project Structure

```
staffing_copilot/
├── src/
│   ├── generate_data.py         # Synthetic data generator (Week 1)
│   ├── embed_employees.py       # Employee embedding pipeline (Week 2)
│   ├── embed_projects.py        # Project embedding pipeline (Week 2)
│   ├── vector_store.py          # FAISS vector store (Week 2)
│   ├── matcher.py               # Combined match scorer (Week 2)
│   ├── optimize_staffing.py     # OR-Tools CP-SAT optimizer (Week 3)
│   ├── retrieve_context.py      # RAG retrieval layer (Week 4)
│   ├── generate_explanation.py  # Ollama LLM explanation (Week 4)
│   └── dashboard.py             # Streamlit dashboard (Week 4)
│
├── notebooks/
│   ├── 01_scope.ipynb
│   ├── 02_pandas.ipynb
│   ├── 03_skill_taxonomy.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_embeddings_intro.ipynb
│   ├── 06_vector_store.ipynb
│   ├── 07_clustering.ipynb
│   ├── 08_cluster_validation.ipynb
│   ├── 09_matcher_testing.ipynb
│   ├── 10_optimizer_intro.ipynb
│   ├── 11_optimizer_testing.ipynb
│   ├── 12_rag_intro.ipynb
│   ├── 13_shap.ipynb
│   └── 14_integration_test.ipynb
│
├── data/
│   └── processed/               # All generated artifacts (16 files)
│
├── tests/
│   ├── test_data_generator.py   # 12 unit tests
│   ├── test_matcher.py          # 11 unit tests
│   └── test_optimizer.py        # 8 unit tests
│
├── docs/
│   └── project_summary.md       # This file
│
└── run_dashboard.bat            # One-click dashboard launcher
```

---

## Full Pipeline — Run Order

```bash
# 1. Generate synthetic data
python src/generate_data.py

# 2. Build embeddings
python src/embed_employees.py
python src/embed_projects.py

# 3. Build vector store
python src/vector_store.py

# 4. Score all candidates
python src/matcher.py

# 5. Optimize staffing
python src/optimize_staffing.py

# 6. Test retriever
python src/retrieve_context.py

# 7. Test explanations
python src/generate_explanation.py

# 8. Launch dashboard
venv\Scripts\python.exe -m streamlit run src/dashboard.py
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Employees | 80 |
| Projects | 30 |
| Role-slots scored | 6,800 (30 × 80 × ~2.8 roles avg) |
| Projects staffed (demo) | 5 (P001–P005) |
| Assignments made | 14 |
| Optimizer status | OPTIMAL |
| Double-bookings | 0 |
| Average match score | 0.810 |
| pytest results | 31 passed, 0 failed |
| Dominant SHAP feature | semantic_score (weight 0.70) |

---

## Key Design Decisions

**1. Synthetic data over real data**
Company staffing data is confidential and unavailable publicly.
Synthetic data generated with Faker (Indian locale, realistic
skill correlations) is a legitimate and standard approach for
proof-of-concept AI systems.

**2. Local LLM (Ollama) over paid API**
Ollama running llama3.2 locally gives zero API cost, offline
capability, and no data privacy concerns when processing employee
information. Explanations are grounded in retrieved facts via RAG —
not hallucinated — so model quality differences matter less.
For the hosted Streamlit Cloud demo — which has no local Ollama
server available — the same explanation function falls back
automatically to Groq's free-tier API (same open-weight Llama
family), so the live demo works without any paid infrastructure.

**3. CP-SAT solver over greedy assignment**
Greedy assignment (give each role its top-scoring candidate
independently) double-books strong candidates across multiple
projects. CP-SAT finds the globally optimal conflict-free plan —
the same mathematical framework used in logistics and scheduling.

**4. Two explainability layers**
Most systems pick one. This project deliberately uses both:
- **SHAP** answers "which numeric features drove this score?" —
  aimed at technical reviewers and data teams
- **LLM text** answers "why this person, in plain English?" —
  aimed at non-technical project managers

**5. Agile + CRISP-DM methodology**
Weekly sprints with demoable deliverables at the end of each week.
CRISP-DM (Business Understanding → Data → Modeling → Evaluation)
applied iteratively within each sprint. No Waterfall — AI
development is inherently exploratory.

---

## Caveats & Limitations

- **Synthetic data** — results demonstrate the pipeline correctly
  but do not reflect real hiring outcomes
- **General-purpose embeddings** — MiniLM was not fine-tuned for
  technical hiring; a domain-specific model would improve matching
- **Full-time assignment assumed** — the optimizer assigns one
  person per role full-time; partial allocations (e.g. 50% on two
  projects) would require extending the capacity constraints
- **Single-run optimization** — the current optimizer staffs a
  fixed set of projects in one call; a production system would
  re-optimize incrementally as new projects arrive
- **Environment-dependent LLM backend** — local runs use Ollama;
  the deployed demo uses a Groq API fallback since cloud hosting
  has no local model server. Both are free; only the backend differs.

---

## Software Development Model

**Agile (sprint-based) + CRISP-DM**, not Waterfall.

| Week | Sprint Goal | Delivered |
|------|-------------|-----------|
| 1 | Believable synthetic dataset | generate_data.py, EDA notebook |
| 2 | Ranked scored shortlist from requirements | embeddings, FAISS, matcher |
| 3 | Conflict-free staffing plan | OR-Tools optimizer |
| 4 | End-to-end demo with explanations | RAG, LLM, SHAP, dashboard |
| day 29–30 | Integration test + documentation | test suite, README, this doc |

---

## Post-Internship Extensions (v2.0)

Everything below was built after the 30-day internship deliverable
above; the results, tech stack, and structure sections above describe
that original snapshot and are left unchanged as a historical record.
Full commit-level detail is in [`../CHANGELOG.md`](../CHANGELOG.md).

### What was added

- **Custom employee intake** — add an employee via a manual form or
  by uploading a CV (PDF/DOCX/TXT), parsed through the same
  Ollama/Groq LLM pipeline `generate_explanation.py` already used,
  then validated against `skills_taxonomy.csv` and reviewed by a
  human before being committed. Assigned a CE0xx id that can never
  collide with the real E0xx range.
- **Custom project intake** — create an ad-hoc project through the
  dashboard, with a capacity pre-check (how many eligible, available
  people exist for each required role) shown before running the
  matcher/solver, so a hopeless request fails fast and cheaply rather
  than after a full CP-SAT solve.
- **Merged-pool matching** — the hardest piece. `Matcher.match()` and
  `match_adhoc()` now take an optional merged `employees_df`
  (real roster + session-added custom employees). Any employee not
  already in the precomputed embedding matrix gets embedded on the
  fly, from a profile string built the same way the original
  `embed_employees.py` pipeline built one — so a custom employee is
  scored in the same semantic space, not bolted on as a second-class
  candidate. This makes custom employees real candidates for
  `staff_custom_project()`'s CP-SAT solve, not just stored records.
- **Duplicate detection** — soft (never-blocking) exact and fuzzy
  name matching against both the real roster and other custom
  employees, so re-uploading the same person's resume gets flagged
  for human confirmation instead of silently creating a second
  record.
- **Dashboard visual pass** — card-based layout for Project Details
  and Assigned Team, color-coded priority/budget badges, collapsible
  per-role candidate-pool sections.
- **Test suite** — grown from 31 to 95 tests, adding coverage for the
  entire storage layer (`employee_store.py`, `project_store.py`) and
  the pure-logic half of `resume_parser.py` (skill/role suggestion,
  duplicate detection), neither of which had any test coverage
  before this pass.

### Design decisions specific to this phase

**Session-state storage, deliberately not "real" persistence yet.**
Every new record — custom employees and projects — lives in
`st.session_state`, gone on reload. This was a deliberate sequencing
choice: build and verify the intake → matching → solving pipeline
first, then decide on a persistence backend once the shape of what
needs persisting is fully known, rather than guessing a schema
upfront. Every call site already reads through `employee_store.py` /
`project_store.py`, so this is meant to be a swap of internals, not a
rewrite of callers, when that decision is made.

**Merging at the data layer, not duplicating logic per call site.**
`employee_store.load_all_employees()` / `project_store.load_all_projects()`
are the single places that combine real + custom records. Every
display and matching call site reads through these instead of each
inventing its own "check custom, then check real" logic — the
alternative (thread a custom-employee check into every place that
touches the roster) would have meant the same merge bug being
possible to introduce in N different places instead of one.

### Known limitations carried into v2.0

- Persistence is in-memory/session-only (see above) — restarting the
  app loses every custom employee and project.
- Premade projects' "Full Candidate Pool" table still reads the
  offline `score_matrix.csv` rather than live-matching, so a custom
  employee is eligible for custom-project solves but won't appear in
  that specific table. Making it live would mean replacing a static
  file read with a per-render matcher call — a larger change than
  what shipped here, and intentionally deferred.
- Role-aware skill validation (flagging a skill that doesn't fit the
  selected role, using `skills_taxonomy.csv`'s `related_roles`
  column) is scoped but not yet built.

---

## Post-v2.0 Extensions (v2.1)

Everything below closes the gap v2.0 left open: custom projects were
real candidates for matching and solving, but didn't get the
explanation/runner-up experience premade projects had, and could
only be solved one at a time. Full commit-level detail is in
[`../CHANGELOG.md`](../CHANGELOG.md).

### What was added

- **Candidate pool persistence** — `project_store.save_candidate_pool()`
  / `get_candidate_pool()`, called right after matching and before
  the CP-SAT solve runs in `staff_custom_project()`. Without this,
  there was nothing for a runner-up panel to read for a custom
  project once the solve finished.
- **`ContextRetriever.retrieve_adhoc()`** — the custom-project
  counterpart to `retrieve()`, built from the in-memory project and
  candidate pool instead of the offline CSV/JSON files that only
  cover the 30 premade projects. Deliberately built to the same
  output shape as `retrieve()`, so `generate_explanation.build_prompt()`
  needed zero changes to consume either one.
- **Dashboard wiring** — the "Role Details & Explanations" panel now
  calls `retrieve_adhoc()` for custom projects instead of showing a
  static "not available" caption, giving them the same explanation
  and runner-up experience premade projects always had.
- **`exclude_ids` in `StaffingOptimizer`** — lets a caller drop
  specific candidates before solving. This is what makes Simulate
  mode's batch solving meaningful: without it, two projects solved
  "together" could still independently pick the same person.
- **Simulate mode** — batch-solve a chosen set of custom projects,
  either jointly (one CP-SAT solve sees all of them at once) or
  staggered (solved one at a time, excluding whoever the previous
  project already took). The two modes are directly comparable on
  average match score, which is what makes the value of joint
  planning demonstrable rather than asserted.
- **Test suite** — grown from 95 to 116 tests: 7 covering
  `exclude_ids` behavior (including the "full exclude of a role's
  only candidates" edge case, which now returns a clean `INFEASIBLE`
  instead of crashing), 6 covering the candidate-pool round-trip,
  and a new `test_retrieve_context.py` (11 tests) whose specific job
  is catching schema drift between `retrieve()` and `retrieve_adhoc()`
  before it reaches `generate_explanation.py`.

### Design decisions specific to this phase

**Schema parity over a shared implementation.** `retrieve()` and
`retrieve_adhoc()` are two separate functions reading from two
different data sources (offline files vs. in-memory session state),
not one function with a branch. What they share is a contract: same
output keys, same nesting, same error shape. `test_retrieve_context.py`
tests that contract directly, since a silent mismatch there would
only surface downstream as a confusing `generate_explanation.py`
failure, not an obvious one.

**Whole-batch failure is intentional, not a bug.** If a Simulate
batch hits a role with zero eligible candidates left, the entire
batch fails rather than silently skipping that one role. This
mirrors real staffing: a plan that quietly drops a role isn't a
smaller version of the plan, it's a different plan the manager never
approved. Failed batches stay easy to retry with a smaller
combination.

### Known limitations carried into v2.1

- Premade projects' "Full Candidate Pool" table is unchanged —
  still reads the offline `score_matrix.csv` only, out of scope for
  this phase.
- Persistence is still `st.session_state`-only, same deferred
  decision as v2.0. **Closed in v3.0, below.**

---

## Post-v2.1 Extensions (v3.0)

Everything below closes v2.1's biggest carried-over limitation:
persistence was still `st.session_state`-only, so every custom
employee and project vanished on refresh. v3.0 adds real accounts and
moves that data into Supabase Postgres, with Row-Level Security as
the actual security boundary rather than an app-side check. Full
commit-level detail is in [`../CHANGELOG.md`](../CHANGELOG.md); the
original item-by-item plan is in `../v3_roadmap.md`.

### What was added

- **Schema + Row-Level Security** — `employees`, `projects`,
  `assignments` tables, each with a nullable `owner_id`: `NULL` means
  shared demo data visible to every account, `owner_id = auth.uid()`
  means private to that user. RLS policies enforce this at the
  database level, not as a Streamlit-side `if` check the anon key
  could bypass. Postgres `CHECK` constraints handle the input
  validation Streamlit alone can't be trusted to enforce, since the
  anon key is public.
- **Login / signup / password reset / email confirmation**
  (`src/auth.py`), backed by Supabase Auth (GoTrue) — password
  hashing and rate-limiting handled server-side, not rolled by hand.
  "Remember me" persists a session across a browser refresh via a
  refresh-token cookie, closing item 22's original known gap. Login
  and password-reset responses are deliberately generic/uniform
  regardless of whether the account exists, to avoid leaking which
  emails are registered.
- **Store internals swapped to Supabase** — `employee_store.py` /
  `project_store.py` now read/write Postgres instead of
  `st.session_state`, with every function signature kept identical.
  `dashboard.py`, `matcher.py`, and `optimize_staffing.py` needed
  zero changes — this was the exact seam Phase 3 (v1.0) was built to
  leave open, now actually exercised.
- **Match scores and candidate pools moved into Postgres too** — a
  `final_score` column on `assignments` and a `candidate_pool jsonb`
  column on `projects`, so a custom project's runner-up detail and
  Full Candidate Pool table now survive a refresh, not just the
  employee/project records themselves.
- **RLS boundary tested against a real Supabase project**, not
  mocked — the one place a mock would have hidden the actual risk.
  Two real test users confirm cross-user reads/writes fail, a spoofed
  `owner_id` insert is rejected, and a regular user can't write a
  `NULL`-owner ("demo") row.
- **"🎬 Demo Mode"** replaces the old session-only "Simulate" mode,
  explicitly scoped to the shared demo roster so it needs no account
  and never touches a logged-in user's own data. **"👥 My Employees"**
  adds search/filter/delete over a user's saved employees. "End
  Project" now fully removes a project instead of only clearing its
  assignments.
- **Test suite** grown from 116 to 156 tests.

### Design decisions specific to this phase

**Option B: migrate only the custom layer, not the demo roster.** The
80-employee/30-project synthetic dataset and its precomputed
embeddings stay exactly as CSV/`.npy` files — a shared, read-only
demo roster every account matches against. Only user-added records
move into Postgres. This kept the tested embeddings pipeline
completely out of the blast radius of a database migration, at the
cost of two data sources instead of one (a real trade-off, not a free
lunch — `load_all_employees()` / `load_own_employees()` are what
absorb that complexity so no caller has to know about it).

**A surrogate primary key, not a natural one.** The schema initially
made `employee_id`/`project_id` (the app-facing `CE0xx`/`C0xx` ids)
the sole primary key. RLS means each user only ever sees their own
rows when computing "next id," so two different users would both
independently land on `CE001` — a real multi-tenant bug that only
shows up with more than one real account, which is exactly why
item 24's real-project RLS test (not the mock) mattered. Fixed with a
surrogate `uuid` primary key plus a composite `(id, owner_id)`
uniqueness constraint instead.

**`token_hash`, not Supabase's default confirmation-link shape.**
Streamlit is server-rendered, so it can only ever read URL query
params — never a `#fragment`, which never reaches the server at all.
That rules out Supabase's implicit link shape outright, and its PKCE
link shape (`?code=...`) needs a `code_verifier` stored client-side at
send time, which a single shared `st.cache_resource` client serving
every visitor on the process can't safely scope per-browser. Both
email templates instead send `token_hash` + `type`, verified
statelessly server-side via `auth.verify_otp()` — no local state to
lose or collide on. See `src/auth.py`'s module docstring for the full
reasoning.

**CAPTCHA scoped, attempted three times, then deliberately dropped.**
The original v3 roadmap called for CAPTCHA on login/signup. Every
implementation attempt (reload+query-param, postMessage+`st.html`, a
split visible/invisible-iframe version) hit the same structural wall:
Streamlit runs third-party JS in a sandboxed iframe with no supported
synchronous channel back to Python, so each workaround just relocated
the same race condition. Rather than patch it a fourth time, CAPTCHA
enforcement was turned off in Supabase's dashboard and Supabase's own
built-in per-IP/per-email rate limiting is the bot-protection baseline
for now — a deliberate, documented trade, not an oversight.

### Known limitations carried into v3.0

- The shared demo roster deliberately stays outside Postgres (Option
  B, above) — a full migration (Option A) is a possible v4+ move.
- Free-tier Supabase projects auto-pause after 7 days without API
  traffic; no keep-alive cron job exists yet, so first load after a
  long idle period may take a moment to resume.
- Real CAPTCHA enforcement is scoped but not built (see above) — the
  robust path, if revisited, is a separate top-level webpage hosting
  Turnstile normally (not inside a Streamlit iframe), redirecting
  back with a session token in the URL.
- GitHub repo tracking (commits/PRs mapped to assigned employees),
  scoped out of v3 from the start, remains deferred to v4 — it's a
  different problem domain (GitHub OAuth, identity mapping, API rate
  limits, and what "work done" should even measure) that needs its
  own design pass rather than being rushed alongside the login
  system.