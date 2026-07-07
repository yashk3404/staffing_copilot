# src/generate_explanation.py
"""
LLM explanation layer using Ollama (free, local) with a Groq API
fallback (also free) for when Ollama isn't reachable — e.g. when this
app is deployed to Streamlit Community Cloud, which has no local
Ollama server.

Takes a retrieved context dict, builds a grounded prompt,
calls the local llama3.2 model (or Groq's hosted Llama if Ollama
is unavailable), returns a plain-English explanation.

Local setup (unchanged):
    ollama pull llama3.2   (run once in terminal)
    ollama serve           (starts automatically on Windows after install)

Cloud fallback setup (optional, only needed for deployment):
    1. Sign up free at https://console.groq.com (no credit card)
    2. Create an API key
    3. Set GROQ_API_KEY as a Streamlit Cloud secret (see dashboard.py),
       or in a local .env file if you want to test the fallback locally

Run:
    python src/generate_explanation.py
"""

import os
import requests
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3.2"

GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.1-8b-instant"


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


def _call_ollama(prompt: str, model: str) -> str:
    """Call the local Ollama server. Raises requests.exceptions.ConnectionError
    if Ollama isn't running, so the caller can fall back to Groq."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 300,
            }
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _call_groq(prompt: str, model: str = GROQ_MODEL) -> str:
    """
    Call Groq's free, OpenAI-compatible API as a fallback when Ollama
    isn't reachable (e.g. on Streamlit Cloud). Requires GROQ_API_KEY
    to be set as an environment variable.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 300,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def generate_explanation(ctx: dict,
                          model: str = OLLAMA_MODEL) -> str:
    """
    Try local Ollama first (used when developing locally).
    If Ollama isn't reachable, fall back to the free Groq API
    (used automatically when deployed, e.g. on Streamlit Cloud).
    Returns an error string (never raises) so the dashboard stays stable.
    """
    if "error" in ctx:
        return f" Cannot explain: {ctx['error']}"

    prompt = build_prompt(ctx)

    try:
        return _call_ollama(prompt, model)

    except requests.exceptions.ConnectionError:
        # Ollama not running — try the free Groq fallback instead
        try:
            return _call_groq(prompt) + "\n\n*(via Groq — Ollama unavailable in this environment)*"
        except RuntimeError:
            return (" Ollama is not running, and no GROQ_API_KEY is "
                    "configured for the fallback. Start Ollama locally "
                    "with 'ollama serve', or set GROQ_API_KEY to enable "
                    "the cloud fallback.")
        except Exception as e:
            return f" Error calling Groq fallback: {str(e)}"

    except Exception as e:
        return f" Error calling Ollama: {str(e)}"


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent))
    from retrieve_context import ContextRetriever

    BASE = Path(__file__).parent.parent / "data" / "processed"

    print("\n── Explanation Generator (Ollama / Groq fallback) ──\n")
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