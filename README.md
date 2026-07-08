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
6. **Presents** everything through an interactive Streamlit dashboard

## Tech stack

| Layer | Technology |
|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Vector store | FAISS (`IndexFlatIP`, cosine similarity) |
| Clustering | KMeans (scikit-learn) |
| Optimization | Google OR-Tools (CP-SAT solver) |
| RAG retrieval | Custom `ContextRetriever` |
| LLM | Ollama (`llama3.2`, local) + Groq (`llama-3.1-8b-instant`, cloud fallback) |
| Explainability | SHAP (LinearExplainer) |
| Dashboard | Streamlit |
| Data | Synthetic (Faker, Indian locale — 80 employees, 30 projects) |
| Testing | pytest (31 tests) |

## Project structure

```
staffing_copilot/
├── src/
│   ├── generate_data.py         # Synthetic data generator
│   ├── embed_employees.py       # Employee embedding pipeline
│   ├── embed_projects.py        # Project embedding pipeline
│   ├── vector_store.py          # FAISS vector store
│   ├── matcher.py               # Combined match scorer
│   ├── optimize_staffing.py     # OR-Tools CP-SAT optimizer
│   ├── retrieve_context.py      # RAG retrieval layer
│   ├── generate_explanation.py  # Ollama LLM explanation
│   └── dashboard.py             # Streamlit dashboard
├── notebooks/                   # Week-by-week build notebooks (01–14)
├── data/processed/              # Generated artifacts (embeddings, scores, plans)
├── tests/                       # pytest suite (31 tests)
├── docs/project_summary.md      # Full project write-up
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
| Employees / Projects | 80 / 30 |
| Role-slots scored | 6,800 |
| Optimizer status | OPTIMAL, zero double-bookings |
| Average match score (demo run) | 0.810 |
| Dominant SHAP feature | `semantic_score` (weight ≈ 0.70) |
| Tests passing | 31 / 31 |

## Caveats

- Data is synthetic — this demonstrates the pipeline, not real hiring outcomes.
- Embeddings use a general-purpose model, not one fine-tuned for technical hiring.
- The optimizer assumes full-time assignment per role; partial allocations would need extended capacity constraints.

Full design rationale (why CP-SAT over greedy, why a local LLM, why two explainability layers) is in [`docs/project_summary.md`](docs/project_summary.md).



## Status

Originally built as a 30-day internship deliverable (see `v1.0-internship`
tag). Actively extended since — see [CHANGELOG.md](CHANGELOG.md) for
what's been added post-internship.