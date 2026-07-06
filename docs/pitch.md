# Staffing Copilot — The Pitch

## The problem

Staffing a project usually comes down to a manager scrolling through spreadsheets and Slack messages, trying to remember who's free, who has the right skills, and who's already been promised to another project. This doesn't scale past a handful of people, and it quietly causes two expensive mistakes: the same strong performer gets penciled into two projects at once, and less-visible employees with the right skills get overlooked simply because nobody thought of them.

## The solution

Staffing Copilot automates the matching and assignment process end to end:

- It reads every employee's skills, experience, and availability, and every project's role requirements, and computes a **semantic match score** for each employee–role pair — not just keyword matching, but an understanding of what skills are actually related.
- It then solves the **assignment problem across all projects simultaneously**, using a constraint solver rather than a greedy "give each role its top candidate" approach. That's what actually prevents double-booking: the system finds the one plan that maximizes overall fit while guaranteeing nobody is assigned twice.
- For every assignment, it generates a **plain-English explanation** — grounded in the real facts of the match, not a hallucinated justification — so a manager can see *why* a person was chosen, not just a score.
- Everything is surfaced in an **interactive dashboard**, so staffing a project becomes a few clicks instead of a spreadsheet exercise.

## Why it's not just "another matching script"

Most matching tools stop at a ranked list. Staffing Copilot goes three steps further: it guarantees the ranked list translates into a *feasible* plan (no conflicts), it explains its own decisions in two different ways — a numeric feature breakdown (SHAP) for technical reviewers and a plain-English narrative (local LLM) for managers — and it runs the explanation layer entirely on a local model, so there's no per-query API cost and no employee data ever leaves the machine.

## Results (proof-of-concept run)

On a synthetic dataset of 80 employees and 30 projects (6,800 scored role-slots), the system produced a conflict-free staffing plan with an average match score of 0.81 and zero double-bookings, backed by a 31-test suite covering the data, the scorer, and the optimizer.

## Who this is for

Anyone running project-based staffing at a scale where "just remember who's free" stops working — internal mobility teams, staffing agencies, or engineering managers juggling more projects than they can track in their head.

For the technical build — embeddings, vector search, the optimization model, and the full architecture — see [`project_summary.md`](project_summary.md).