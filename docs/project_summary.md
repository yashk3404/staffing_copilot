# Staffing Copilot — Project Summary

**Built by:** Kumar Yash
**Role:** Python AI Intern, Yash Technologies  
**Duration:** 30 days  
**Type:** End-to-end AI system (proof of concept)

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
| day 29–30 | Integration test + documentation 