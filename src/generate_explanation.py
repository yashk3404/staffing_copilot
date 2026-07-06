# src/generate_explanation.py
"""
LLM explanation layer using Ollama (free, local).

Takes a retrieved context dict, builds a grounded prompt,
calls the local llama3.2 model, returns plain-English explanation.

Prerequisites:
    ollama pull llama3.2   (run once in terminal)
    ollama serve           (starts automatically on Windows after install)

Run:
    python src/generate_explanation.py
"""

import requests
import json
from pathlib import Path


OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"


def build_prompt(ctx: dict) -> str:
    """Format a retrieved context dict into a grounded LLM prompt."""
    assigned = ctx["assigned"]
    project  = ctx["project"]
    ru       = ctx["runner_up"]

    runner_up_section = ""
    if ru:
        direction = "higher" if ru["score_gap"] > 0 else "lower"
        runner_up_section = f"""
Runner-up considered:
- Name: {ru['name']}
- Job title: {ru['actual_role']}
- Experience: {ru['experience_years']} years
- Availability: {ru['availability_pct']}%
- Match score: {ru['score']} ({abs(ru['score_gap']):.4f} {direction} than assigned)
"""

    return f"""You are a staffing coordinator writing a brief explanation for a project manager.
Explain the staffing decision below in exactly 3-4 sentences.
Use the actual names and numbers provided. Do not invent any facts not listed here.

PROJECT: {project['name']} ({ctx['project_id']})
ROLE TO FILL: {ctx['role']}
PROJECT SUMMARY: {project['summary'][:300]}

ASSIGNED CANDIDATE:
- Name: {assigned['name']}
- Job title: {assigned['actual_role']}
- Experience: {assigned['experience_years']} years
- Availability: {assigned['availability_pct']}%
- Match score: {assigned['score']} out of 1.0
- Skills: {assigned['profile'][:200]}
{runner_up_section}
Write a plain-English explanation of why {assigned['name']} was selected
for the {ctx['role']} role on {project['name']}.
If a runner-up is listed, explain in one sentence why they were not chosen."""


def generate_explanation(ctx: dict,
                          model: str = DEFAULT_MODEL) -> str:
    """
    Call local Ollama model and return the explanation text.
    Returns an error string (never raises) so the dashboard stays stable.
    """
    if "error" in ctx:
        return f"⚠️ Cannot explain: {ctx['error']}"

    prompt = build_prompt(ctx)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,   # low = more factual, less creative
                    "num_predict": 300,   # max tokens to generate
                }
            },
            timeout=120   # llama3.2 is fast but give it room
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return ("⚠️ Ollama is not running. "
                "Start it by running 'ollama serve' in a terminal.")
    except Exception as e:
        return f"⚠️ Error calling Ollama: {str(e)}"


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent))
    from retrieve_context import ContextRetriever

    BASE = Path(__file__).parent.parent / "data" / "processed"

    print("\n── Explanation Generator (Ollama) ─────────────────")
    retriever = ContextRetriever(str(BASE))

    test_slots = [
        ("P001", "Backend Dev"),
        ("P002", "Android Dev"),
        ("P004", "Data Scientist"),
    ]

    for project_id, role in test_slots:
        ctx = retriever.retrieve(project_id, role)
        print(f"\n{'='*60}")
        print(f"Project : {ctx['project']['name']} | Role: {role}")
        print(f"Assigned: {ctx['assigned']['name']} "
              f"(score {ctx['assigned']['score']})")
        print("-" * 60)
        explanation = generate_explanation(ctx)
        print(explanation)

    print("\n── Done ────────────────────────────────────────────\n")