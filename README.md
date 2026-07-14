# Staffing Copilot

An AI-powered employee-to-project matching and staffing optimization system, now with real accounts and persistence. It semantically matches employees to project roles, solves conflict-free assignment across multiple projects using constrained optimization, and explains every staffing decision in plain English using an LLM — with a Streamlit dashboard as the front end and Supabase handling auth and storage for anything a user creates.

Built as a 30-day, end-to-end proof-of-concept during a Python AI internship, then actively extended post-internship. See [`docs/project_summary.md`](docs/project_summary.md) for the full write-up (design decisions, results, caveats).

## Live Demo

Try it here: **[staffing-copilot](https://staffing-copilot.streamlit.app/)**

Sign up with any email to try it — your own employees/projects are private to your account and don't affect the shared demo roster. Or skip signing up entirely and explore **🎬 Demo Mode**, which needs no account.

## What it does

1. **Semantically matches** employees to project roles using sentence embeddings + cosine similarity
2. **Clusters** employees by skill profile to validate embedding quality
3. **Optimally assigns** employees across multiple projects at once using a CP-SAT constraint solver — no double-booking, no conflicts
4. **Explains** every staffing decision in plain English using an LLM (local Ollama for dev, free Groq API fallback for the hosted demo)
5. **Shows** which numeric features drove each match score using SHAP
6. **Accounts, backed by Supabase Auth** — sign up, log in, "remember me" session persistence across page refreshes, email confirmation, and a self-service password reset flow, all enforced server-side (hashing, rate-limiting) rather than hand-rolled
7. **Takes custom projects and employees, and actually keeps them** — create an ad-hoc project or add an employee (manual form or CV upload, parsed via the same LLM pipeline), and it's saved to your account in Postgres, not just `st.session_state` — it survives a refresh, a new session, or logging in from another device
8. **Keeps every account's data private by construction** — Postgres Row-Level Security enforces "you can only see your own rows" at the database level, not as an app-side `if` check that a bug could bypass
9. **🎬 Demo Mode** gives anyone (logged in or not) the full experience against the built-in 80-employee / 30-project synthetic roster, including batch-solving several premade projects together to compare joint vs. staggered staffing outcomes
10. **Gives custom projects the same live explanations and runner-up detail as premade ones**, with match scores and the full ranked candidate pool persisted per project so they don't go blank on a refresh
11. **Presents** everything through an interactive Streamlit dashboard: 🎬 Demo Mode / 📂 Browse Projects / ➕ Create Project / 👥 My Employees / 🧑‍💻 Add Employee

## Tech stack

| Layer | Technology |
|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Vector store | FAISS (`IndexFlatIP`, cosine similarity) — offline pipeline only, not imported by the deployed app |
| Clustering | KMeans (scikit-learn) |
| Optimization | Google OR-Tools (CP-SAT solver) |
| RAG retrieval | Custom `ContextRetriever` (premade + ad-hoc) |
| LLM | Ollama (`llama3.2`, local) + Groq (`llama-3.1-8b-instant`, cloud fallback via plain `requests`, no SDK) |
| Resume/CV parsing | pdfplumber (PDF), python-docx (DOCX) + the same Ollama/Groq LLM pipeline |
| Explainability | SHAP (LinearExplainer) — offline notebook step; dashboard just displays the resulting static images |
| Auth | Supabase Auth (GoTrue) — email/password, email confirmation, password reset, Postgres-native rate limiting |
| Database | Supabase Postgres with Row-Level Security — stores every account's own employees/projects/assignments |
| Session persistence | `streamlit-cookies-controller` — "remember me" survives a browser refresh |
| Dashboard | Streamlit |
| Data | Synthetic shared demo roster (Faker, Indian locale — 80 employees, 30 projects, `owner_id IS NULL`) + per-user custom employees/projects in Postgres |
| Testing | pytest (156 tests) |

## Project structure

```
staffing_copilot/
├── src/
│   ├── generate_data.py         # Synthetic data generator
│   ├── embed_employees.py       # Employee embedding pipeline
│   ├── embed_projects.py        # Project embedding pipeline
│   ├── vector_store.py          # FAISS vector store (offline)
│   ├── matcher.py               # Combined match scorer (demo + merged custom pool)
│   ├── optimize_staffing.py     # OR-Tools CP-SAT optimizer (premade + ad-hoc)
│   ├── retrieve_context.py      # RAG retrieval layer (premade + ad-hoc)
│   ├── generate_explanation.py  # Ollama/Groq LLM explanation
│   ├── auth.py                  # Supabase Auth — login/signup, remember-me, password reset, email confirmation
│   ├── employee_store.py        # Custom (CE0xx) employee CRUD, backed by Supabase
│   ├── project_store.py         # Custom (C0xx) project CRUD + candidate pool persistence, backed by Supabase
│   ├── resume_parser.py         # CV/resume text extraction + LLM structuring
│   └── dashboard.py             # Streamlit dashboard
├── supabase/
│   └── migrations/              # Schema, RLS policies, and follow-up migrations (run in order in the Supabase SQL editor)
├── scripts/
│   └── migrate_demo_roster_to_user.py  # One-time: import the 80-employee demo roster into your own account
├── notebooks/                    # Week-by-week build notebooks (01–14)
├── data/processed/              # Generated artifacts (embeddings, scores, plans)
├── tests/                        # pytest suite (156 tests)
├── docs/
│   ├── project_summary.md       # Full project write-up
│   └── pitch.md                 # One-page pitch
├── CHANGELOG.md                  # Full v1.0-internship → v3.0 history
├── requirements.txt               # Deployed-app dependencies only
└── requirements-dev.txt           # + notebooks, SHAP plotting, data generation, pytest
```

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1          # Windows
# source venv/bin/activate         # macOS/Linux

pip install -r requirements-dev.txt   # includes requirements.txt + notebook/test/plotting deps
```

You'll also need [Ollama](https://ollama.com) installed locally with the `llama3.2` model pulled:
```bash
ollama pull llama3.2
```
> **Note:** The app defaults to a local Ollama model. On the hosted
> Streamlit Cloud demo above, there's no local Ollama server, so it
> automatically falls back to Groq's free API for explanations instead.
> To test that fallback locally, add a `GROQ_API_KEY` to
> `.streamlit/secrets.toml` (get a free key at
> [console.groq.com](https://console.groq.com)).

### Supabase setup (required for auth/persistence)

1. Create a free project at [supabase.com](https://supabase.com).
2. In the SQL Editor, run each file in `supabase/migrations/` **in order** (`0001`, `0002`, `0003`), reading each one before running it — none are destructive to existing data, but review the practice anyway.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in:
   ```toml
   SUPABASE_URL = "https://YOUR-PROJECT-REF.supabase.co"
   SUPABASE_KEY = "your-anon-or-publishable-key"   # anon/publishable key, NEVER service_role
   APP_URL = "http://localhost:8501"                # or your deployed URL, e.g. https://staffing-copilot.streamlit.app
   ```
   On Streamlit Cloud, set these same three keys in the app's **Settings → Secrets** instead of a file. `APP_URL` in particular must be the deployed URL there — left unset, confirmation/reset links fall back to `localhost` and won't be reachable once deployed.
4. **Custom SMTP** — Supabase's built-in mailer is heavily rate-limited and not meant for production sign-ups. Under **Authentication → Emails → SMTP Settings**, enable Custom SMTP with a provider like [Brevo](https://www.brevo.com) (free tier, 300 emails/day):
   - **Host:** `smtp-relay.brevo.com`, **Port:** `587`
   - **Username / Password:** from Brevo → your profile icon → **SMTP & API → SMTP tab**. The password field is a generated **SMTP key**, *not* your Brevo account password.
   - **Sender email:** must exactly match a sender verified in Brevo.
5. **Email templates** — under **Authentication → Emails → Templates**, both **Confirm signup** and **Reset password** need their link changed from Supabase's default `{{ .ConfirmationURL }}` to:
   ```
   {{ .SiteURL }}?token_hash={{ .TokenHash }}&type=signup      <!-- Confirm signup -->
   {{ .SiteURL }}?token_hash={{ .TokenHash }}&type=recovery    <!-- Reset password -->
   ```
   This app is server-rendered (Streamlit), so it can only read query params, never a URL fragment — that rules out Supabase's default implicit (`#access_token=...`) and PKCE (`?code=...`) link shapes, since the latter needs client-side state this app's single shared server process can't safely keep per-visitor. `token_hash` is verified statelessly server-side instead. See `src/auth.py`'s module docstring for the full reasoning.
6. **Site URL / Redirect URLs** — under **Authentication → URL Configuration**, set Site URL to your deployed `APP_URL`, and add it to the Redirect URLs allow-list.
7. If your Supabase project ever holds real demo data you want every account to see, insert it with `owner_id IS NULL` — that's what makes it shared rather than private (see `supabase/migrations/0001_schema_and_rls.sql`).

To debug an email that isn't arriving: check **Supabase Dashboard → Authentication → Logs** (the actual SMTP error, if any, shows up in the `event_message` JSON) and **Brevo Dashboard → Transactional → Email Activity** (confirms whether Supabase even reached Brevo).

## Running the full pipeline

```bash
python src/generate_data.py
python src/embed_employees.py
python src/embed_projects.py
python src/vector_store.py
python src/matcher.py
python src/optimize_staffing.py
python src/retrieve_context.py
python src/generate_explanation.py

# Launch the dashboard
venv\Scripts\python.exe -m streamlit run src/dashboard.py
```

## Running tests

```bash
pytest tests/ -v
```

`tests/test_rls_boundary.py` is the one exception — it deliberately runs against a **real** Supabase test project rather than the mocked test double (a broken RLS policy would look fine against an in-memory fake). It skips itself cleanly if the required env vars (`SUPABASE_URL`, `SUPABASE_KEY`, two test-user email/password pairs) aren't set, so it won't break a normal `pytest tests/` run or CI.

## Key results

| Metric | Value |
|---|---|
| Employees / Projects (shared demo roster) | 80 / 30, plus unlimited per-account custom employees/projects |
| Role-slots scored (demo roster) | 6,800 |
| Optimizer status | OPTIMAL, zero double-bookings |
| Average match score (demo run) | 0.810 |
| Dominant SHAP feature | `semantic_score` (weight ≈ 0.70) |
| Tests passing | 156 / 156 |

## Caveats

- Data is synthetic — this demonstrates the pipeline, not real hiring outcomes.
- Embeddings use a general-purpose model, not one fine-tuned for technical hiring.
- The optimizer assumes full-time assignment per role; partial allocations would need extended capacity constraints.
- The shared demo roster (CSV + precomputed embeddings) is deliberately left as static files rather than migrated into Postgres (Option B in `v3_roadmap.md`) — only user-added records live in the database. A full migration of the demo roster itself is a possible v4+ move, not required for this to work.
- Free-tier Supabase projects auto-pause after 7 days of no API traffic — first load after inactivity may take a moment to resume.

Full design rationale (why CP-SAT over greedy, why a local LLM, why two explainability layers, why Supabase over Clerk/Firebase/Appwrite) is in [`docs/project_summary.md`](docs/project_summary.md).

## Status

Originally built as a 30-day internship deliverable (see `v1.0-internship`
tag). Actively extended since — custom employee/project intake, resume
parsing, and a merged matching pool were added in v2.0; live
explanations and runner-up detail for custom projects, plus a
multi-project Simulate mode, were added in v2.1; real accounts,
Postgres persistence with Row-Level Security, and full session/email
hardening (remember-me, password reset, email confirmation) were
added in v3.0 — see [CHANGELOG.md](CHANGELOG.md) for the full history
of what's changed post-internship.