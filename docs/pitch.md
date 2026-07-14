# Staffing Copilot — The Pitch
**[Live demo →](https://staffing-copilot.streamlit.app/)**

## The problem

Staffing a project usually comes down to a manager scrolling through spreadsheets and Slack messages, trying to remember who's free, who has the right skills, and who's already been promised to another project. This doesn't scale past a handful of people, and it quietly causes two expensive mistakes: the same strong performer gets penciled into two projects at once, and less-visible employees with the right skills get overlooked simply because nobody thought of them.

## The solution

Staffing Copilot automates the matching and assignment process end to end:

- It reads every employee's skills, experience, and availability, and every project's role requirements, and computes a **semantic match score** for each employee–role pair — not just keyword matching, but an understanding of what skills are actually related.
- It then solves the **assignment problem across all projects simultaneously**, using a constraint solver rather than a greedy "give each role its top candidate" approach. That's what actually prevents double-booking: the system finds the one plan that maximizes overall fit while guaranteeing nobody is assigned twice.
- For every assignment, it generates a **plain-English explanation** — grounded in the real facts of the match, not a hallucinated justification — so a manager can see *why* a person was chosen, not just a score.
- It's not limited to a fixed roster: a manager can **add a new employee by uploading their CV** (parsed and structured by the same LLM pipeline, then reviewed by a human before it's committed) or **spin up an ad-hoc project on the spot**, and both are immediately real candidates for matching and solving — not static records bolted onto the side.
- Ad-hoc projects get the same treatment as premade ones: the same **plain-English explanations and runner-up detail**, not a "not available" placeholder. Try the built-in **Demo Mode** to batch-solve several premade projects together and compare that against staggering them one at a time, to see the real cost of staffing reactively instead of planning jointly.
- It's **multi-tenant, with real accounts** — sign up, and everything you add is private to you and actually persists: a page refresh, a new session, or logging in from another device doesn't lose it. That privacy is enforced at the database level (Postgres Row-Level Security), not by an app-side check that a bug could bypass.
- Everything is surfaced in an **interactive dashboard**, so staffing a project becomes a few clicks instead of a spreadsheet exercise.

## Why it's not just "another matching script"

Most matching tools stop at a ranked list. Staffing Copilot goes further: it guarantees the ranked list translates into a *feasible* plan (no conflicts), it explains its own decisions in two different ways — a numeric feature breakdown (SHAP) for technical reviewers and a plain-English narrative (LLM) for managers — it uses free, open-weight LLMs throughout (a local model for development, a free hosted fallback for the live demo), so there's no per-query API cost, and it's built as a real multi-user product (Supabase Auth + Postgres with Row-Level Security) rather than a single-session demo script.

## Results (proof-of-concept run)

On a synthetic dataset of 80 employees and 30 projects (6,800 scored role-slots), the system produced a conflict-free staffing plan with an average match score of 0.81 and zero double-bookings, backed by a 156-test suite covering the data, the scorer, the optimizer, the custom employee/project intake layer, the live-explanation/Demo-Mode retrieval path, and the Row-Level Security boundary itself (tested against a real Supabase project, not mocked).

## Who this is for

Anyone running project-based staffing at a scale where "just remember who's free" stops working — internal mobility teams, staffing agencies, or engineering managers juggling more projects than they can track in their head.

For the technical build — embeddings, vector search, the optimization model, and the full architecture — see [`project_summary.md`](project_summary.md).