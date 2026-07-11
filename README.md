# Staffing Copilot

An AI-powered employee-to-project matching and staffing optimization system. It semantically matches employees to project roles, solves conflict-free assignment across multiple projects using constrained optimization, and explains every staffing decision in plain English using a local LLM — with a Streamlit dashboard as the front end.

Built as a 30-day, end-to-end proof-of-concept during a Python AI internship. See [`docs/project_summary.md`](docs/project_summary.md) for the full write-up (design decisions, results, caveats).

## Live Demo

Try it here: **[staffing-copilot](https://staffing-copilot.streamlit.app/)**

## What it does

1. **Semantically matches** employees to project roles using sentence embeddings + cosine similarity
2. **Clusters** employees by skill profile to validate embedding quality
3. **Optimally assigns** employees across multiple projects at once using a CP-SAT constraint solver — no double-booking, no conflicts
4. **Explains** every staffing decision in plain English using a locally-running LLM (zero API cost)
5. **Shows** which numeric features drove each match score using SHAP
6. **Takes custom projects and employees** — create an ad-hoc project or add an employee (manual form or CV upload, parsed via the same LLM pipeline) mid-session, and both become real candidates in the matcher/solver, not just stored records
7. **Gives custom projects the same live explanations and runner-up detail as premade ones**, and lets you batch-solve several custom projects at once in **Simulate mode** to compare joint vs staggered staffing outcomes
8. **Presents** everything through an interactive Streamlit dashboard

## Tech stack

| Layer | Technology |
|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Vector store | FAISS (`IndexFlatIP`, cosine similarity) |
| Clustering | KMeans (scikit-learn) |
| Optimization | Google OR-Tools (CP-SAT solver) |
| RAG retrieval | Custom `ContextRetriever` |
| LLM | Ollama (`llama3.2`, local) + Groq (`llama-3.1-8b-instant`, cloud fallback) |
| Resume/CV parsing | pdfplumber (PDF), python-docx (DOCX) + the same Ollama/Groq LLM pipeline |
| Explainability | SHAP (LinearExplainer) |
| Dashboard | Streamlit |
| Data | Synthetic (Faker, Indian locale — 80 employees, 30 projects) + session-added custom employees/projects |
| Testing | pytest (116 tests) |

## Project structure

```
staffing_copilot/
├── src/
│   ├── generate_data.py         # Synthetic data generator
│   ├── embed_employees.py       # Employee embedding pipeline
│   ├── embed_projects.py        # Project embedding pipeline
│   ├── vector_store.py          # FAISS vector store
│   ├── matcher.py               # Combined match scorer (premade + merged custom pool)
│   ├── optimize_staffing.py     # OR-Tools CP-SAT optimizer (premade + ad-hoc)
│   ├── retrieve_context.py      # RAG retrieval layer
│   ├── generate_explanation.py  # Ollama/Groq LLM explanation
│   ├── employee_store.py        # Custom (CE0xx) employee intake + merge
│   ├── project_store.py         # Custom (C0xx) project intake + capacity check
│   ├── resume_parser.py         # CV/resume text extraction + LLM structuring
│   └── dashboard.py             # Streamlit dashboard
├── notebooks/                   # Week-by-week build notebooks (01–14)
├── data/processed/              # Generated artifacts (embeddings, scores, plans)
├── tests/                       # pytest suite (116 tests)
├── docs/project_summary.md      # Full project write-up
├── CHANGELOG.md                 # v1.0-internship → v2.1 history
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1          # Windows
# source venv/bin/activate         # macOS/Linux

pip install -r requirements.txt
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

## Key results

| Metric | Value |
|---|---|
| Employees / Projects | 80 / 30 (plus session-added custom employees/projects) |
| Role-slots scored | 6,800 |
| Optimizer status | OPTIMAL, zero double-bookings |
| Average match score (demo run) | 0.810 |
| Dominant SHAP feature | `semantic_score` (weight ≈ 0.70) |
| Tests passing | 116 / 116 |

## Caveats

- Data is synthetic — this demonstrates the pipeline, not real hiring outcomes.
- Embeddings use a general-purpose model, not one fine-tuned for technical hiring.
- The optimizer assumes full-time assignment per role; partial allocations would need extended capacity constraints.
- Custom employees/projects added through the dashboard live in `st.session_state` only — they don't persist across a page reload or a new session. Real persistence is a deliberately deferred decision (see `CHANGELOG.md`); every call site already reads through `employee_store.py`/`project_store.py`, so swapping the internals later shouldn't require touching callers.

Full design rationale (why CP-SAT over greedy, why a local LLM, why two explainability layers) is in [`docs/project_summary.md`](docs/project_summary.md).

## Status

Originally built as a 30-day internship deliverable (see `v1.0-internship`
tag). Actively extended since — custom employee/project intake, resume
parsing, and a merged matching pool were added in v2.0; live
explanations and runner-up detail for custom projects, plus a
multi-project Simulate mode, were added in v2.1 — see
[CHANGELOG.md](CHANGELOG.md) for the full history of what's changed
post-internship.