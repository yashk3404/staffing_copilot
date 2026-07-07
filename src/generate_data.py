# src/generate_data.py
"""
Synthetic data generator for Staffing Copilot.
Produces employees.csv and projects.csv with realistic correlations.

Run:
    python src/generate_data.py
"""

import pandas as pd
import numpy as np
import random
from faker import Faker
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")          # Indian locale → realistic names
Faker.seed(SEED)

# ── Output paths ─────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Skill taxonomy (must match skills_taxonomy.csv) ───────────────
SKILL_TAXONOMY = {
    "Programming":   ["Python", "Java", "JavaScript", "TypeScript"],
    "Frontend":      ["React", "TypeScript"],
    "Backend":       ["Node.js", "FastAPI", "Django", "REST APIs",
                      "GraphQL", "Microservices"],
    "Database":      ["SQL", "PostgreSQL", "MongoDB", "Redis"],
    "Cloud":         ["AWS"],
    "DevOps":        ["Docker", "Kubernetes", "CI/CD", "Terraform", "Linux"],
    "ML/AI":         ["TensorFlow", "PyTorch", "scikit-learn", "NLP",
                      "Computer Vision", "LLMs", "RAG", "FAISS"],
    "Data":          ["pandas", "NumPy", "Spark", "Kafka", "Airflow"],
    "Mobile":        ["Android", "Kotlin", "Swift", "iOS"],
    "Analytics":     ["Power BI", "Tableau", "Excel"],
    "Soft Skills":   ["Project Management", "Agile/Scrum"],
    "Architecture":  ["System Design", "Microservices"],
}

# ── Role definitions with correlated skill pools ──────────────────
ROLE_PROFILES = {
    "Backend Dev": {
        "core_skills":    ["Python", "REST APIs", "PostgreSQL", "Docker"],
        "bonus_skills":   ["FastAPI", "Django", "Redis", "System Design",
                           "Microservices", "AWS", "Node.js", "GraphQL"],
        "cost_band_dist": {"A": 0.3, "B": 0.5, "C": 0.2},
        "exp_range":      (1, 10),
    },
    "Data Engineer": {
        "core_skills":    ["Python", "SQL", "AWS", "pandas"],
        "bonus_skills":   ["Spark", "Kafka", "Airflow", "PostgreSQL",
                           "NumPy", "Docker", "Linux"],
        "cost_band_dist": {"A": 0.2, "B": 0.5, "C": 0.3},
        "exp_range":      (2, 12),
    },
    "ML Engineer": {
        "core_skills":    ["Python", "scikit-learn", "pandas", "NumPy"],
        "bonus_skills":   ["TensorFlow", "PyTorch", "NLP", "Computer Vision",
                           "LLMs", "RAG", "FAISS", "AWS", "Docker"],
        "cost_band_dist": {"A": 0.2, "B": 0.4, "C": 0.4},
        "exp_range":      (2, 10),
    },
    "Frontend Dev": {
        "core_skills":    ["JavaScript", "React", "TypeScript", "CSS"],
        "bonus_skills":   ["Node.js", "GraphQL", "REST APIs", "Git"],
        "cost_band_dist": {"A": 0.35, "B": 0.45, "C": 0.2},
        "exp_range":      (1, 8),
    },
    "Full Stack Dev": {
        "core_skills":    ["JavaScript", "React", "Node.js", "PostgreSQL"],
        "bonus_skills":   ["TypeScript", "REST APIs", "Docker", "AWS",
                           "GraphQL", "MongoDB"],
        "cost_band_dist": {"A": 0.2, "B": 0.5, "C": 0.3},
        "exp_range":      (2, 10),
    },
    "DevOps": {
        "core_skills":    ["Docker", "Kubernetes", "Linux", "CI/CD"],
        "bonus_skills":   ["Terraform", "AWS", "Python", "Kafka",
                           "System Design"],
        "cost_band_dist": {"A": 0.15, "B": 0.45, "C": 0.4},
        "exp_range":      (3, 12),
    },
    "Android Dev": {
        "core_skills":    ["Android", "Kotlin", "Java", "REST APIs"],
        "bonus_skills":   ["Python", "SQL", "Firebase", "Git"],
        "cost_band_dist": {"A": 0.3, "B": 0.5, "C": 0.2},
        "exp_range":      (1, 9),
    },
    "Data Scientist": {
        "core_skills":    ["Python", "scikit-learn", "pandas", "SQL"],
        "bonus_skills":   ["TensorFlow", "PyTorch", "NLP", "Tableau",
                           "Power BI", "NumPy", "AWS"],
        "cost_band_dist": {"A": 0.2, "B": 0.45, "C": 0.35},
        "exp_range":      (2, 10),
    },
    "Project Manager": {
        "core_skills":    ["Project Management", "Agile/Scrum", "Excel"],
        "bonus_skills":   ["Power BI", "Tableau", "SQL", "System Design"],
        "cost_band_dist": {"A": 0.1, "B": 0.4, "C": 0.5},
        "exp_range":      (4, 15),
    },
}

DEPARTMENTS = ["Engineering", "Data & AI", "Mobile", "Platform",
               "Analytics", "Product"]

LOCATIONS = ["Remote", "Bangalore", "Mumbai", "Hyderabad",
             "Pune", "Chennai", "Delhi"]

PROFICIENCY_LEVELS = ["Beginner", "Intermediate", "Expert"]


def pick_cost_band(dist: dict) -> str:
    bands, weights = zip(*dist.items())
    return random.choices(bands, weights=weights)[0]


def assign_proficiency(skill: str, core_skills: list,
                       experience_years: int) -> str:
    """Core skills get higher proficiency; senior employees are more expert."""
    base = 1 if skill in core_skills else 0
    seniority_boost = min(experience_years // 4, 1)
    level_idx = min(base + seniority_boost + random.randint(0, 1), 2)
    return PROFICIENCY_LEVELS[level_idx]


def generate_employees(n: int = 80) -> pd.DataFrame:
    """Generate n synthetic employees with correlated skills."""
    rows = []
    roles = list(ROLE_PROFILES.keys())
    # Weight distribution so Backend Dev and Data Engineer are most common
    role_weights = [0.2, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05, 0.07, 0.03]

    for i in range(1, n + 1):
        role      = random.choices(roles, weights=role_weights)[0]
        profile   = ROLE_PROFILES[role]
        exp_years = random.randint(*profile["exp_range"])
        cost_band = pick_cost_band(profile["cost_band_dist"])

        # Pick core skills (always included) + 1–3 bonus skills
        core   = profile["core_skills"].copy()
        bonus  = random.sample(
            profile["bonus_skills"],
            k=min(random.randint(1, 3), len(profile["bonus_skills"]))
        )
        # Add Git + Agile universally with 80% chance
        universal = [s for s in ["Git", "Agile/Scrum"]
                     if random.random() < 0.8]
        all_skills = list(dict.fromkeys(core + bonus + universal))

        proficiencies = [
            assign_proficiency(s, core, exp_years) for s in all_skills
        ]

        # Availability: seniors tend to be more allocated (less available)
        if cost_band == "C":
            avail = random.choice([40, 50, 60, 60, 80])
        elif cost_band == "B":
            avail = random.choice([60, 80, 80, 100])
        else:
            avail = random.choice([80, 100, 100])

        rows.append({
            "employee_id":      f"E{i:03d}",
            "name":             fake.name(),
            "role":             role,
            "experience_years": exp_years,
            "skills":           ";".join(all_skills),
            "proficiency":      ";".join(proficiencies),
            "availability_pct": avail,
            "cost_band":        cost_band,
            "department":       random.choice(DEPARTMENTS),
            "location":         random.choice(LOCATIONS),
        })

    return pd.DataFrame(rows)


# ── Project generation ────────────────────────────────────────────

PROJECT_TEMPLATES = [
    {
        "name_template":   "FinTech Mobile App",
        "client_pool":     ["HDFC Tech", "Paytm", "PhonePe", "Groww"],
        "required_roles":  ["Backend Dev", "Android Dev", "Data Engineer"],
        "required_skills": ["Python", "Android", "REST APIs", "SQL", "AWS"],
        "min_experience":  3,
        "deadline_range":  (30, 90),
        "budget_band":     "high",
        "priority":        "critical",
    },
    {
        "name_template":   "ML Recommendation Engine",
        "client_pool":     ["Flipkart", "Amazon IN", "Myntra", "Nykaa"],
        "required_roles":  ["ML Engineer", "Data Engineer", "Backend Dev"],
        "required_skills": ["Python", "TensorFlow", "pandas", "SQL", "FastAPI"],
        "min_experience":  4,
        "deadline_range":  (45, 120),
        "budget_band":     "high",
        "priority":        "high",
    },
    {
        "name_template":   "Data Analytics Dashboard",
        "client_pool":     ["Infosys", "TCS", "Wipro", "HCL"],
        "required_roles":  ["Data Scientist", "Frontend Dev", "Data Engineer"],
        "required_skills": ["Python", "SQL", "React", "Power BI", "Tableau"],
        "min_experience":  2,
        "deadline_range":  (20, 60),
        "budget_band":     "medium",
        "priority":        "medium",
    },
    {
        "name_template":   "Cloud Migration",
        "client_pool":     ["BPCL", "ONGC Tech", "Reliance Jio"],
        "required_roles":  ["DevOps", "Backend Dev", "Data Engineer"],
        "required_skills": ["AWS", "Docker", "Kubernetes", "Python", "Terraform"],
        "min_experience":  4,
        "deadline_range":  (60, 150),
        "budget_band":     "high",
        "priority":        "high",
    },
    {
        "name_template":   "E-Commerce Platform",
        "client_pool":     ["Snapdeal", "Meesho", "BigBasket"],
        "required_roles":  ["Full Stack Dev", "Backend Dev", "DevOps"],
        "required_skills": ["React", "Node.js", "PostgreSQL", "Docker", "AWS"],
        "min_experience":  3,
        "deadline_range":  (45, 90),
        "budget_band":     "medium",
        "priority":        "high",
    },
    {
        "name_template":   "RAG Chatbot",
        "client_pool":     ["Zoho", "Freshworks", "InMobi"],
        "required_roles":  ["ML Engineer", "Backend Dev", "Data Engineer"],
        "required_skills": ["Python", "LLMs", "RAG", "FAISS", "FastAPI"],
        "min_experience":  3,
        "deadline_range":  (30, 75),
        "budget_band":     "medium",
        "priority":        "high",
    },
    {
        "name_template":   "iOS Banking App",
        "client_pool":     ["Axis Bank Tech", "Kotak Digital", "SBI Cards"],
        "required_roles":  ["Android Dev", "Backend Dev", "Data Engineer"],
        "required_skills": ["Swift", "iOS", "REST APIs", "PostgreSQL", "AWS"],
        "min_experience":  3,
        "deadline_range":  (60, 120),
        "budget_band":     "high",
        "priority":        "critical",
    },
    {
        "name_template":   "Internal HR Tool",
        "client_pool":     ["Internal", "Yash Technologies"],
        "required_roles":  ["Full Stack Dev", "Data Scientist"],
        "required_skills": ["React", "Python", "SQL", "REST APIs"],
        "min_experience":  2,
        "deadline_range":  (20, 45),
        "budget_band":     "low",
        "priority":        "medium",
    },
]


def generate_projects(n: int = 30) -> pd.DataFrame:
    """Generate n synthetic projects from templates."""
    rows = []
    for i in range(1, n + 1):
        tmpl   = random.choice(PROJECT_TEMPLATES)
        client = random.choice(tmpl["client_pool"])

        # Add a numeric suffix to make names unique
        suffix = f" v{random.randint(1,3)}" if random.random() > 0.5 else ""
        name   = f"{tmpl['name_template']}{suffix} — {client}"

        rows.append({
            "project_id":      f"P{i:03d}",
            "project_name":    name,
            "client":          client,
            "required_roles":  ";".join(tmpl["required_roles"]),
            "required_skills": ";".join(tmpl["required_skills"]),
            "min_experience":  tmpl["min_experience"] + random.randint(-1, 1),
            "deadline_days":   random.randint(*tmpl["deadline_range"]),
            "budget_band":     tmpl["budget_band"],
            "priority":        tmpl["priority"],
        })

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Staffing Copilot — Synthetic Data Generator")
    print("=" * 55)

    print("\n[1/2] Generating 80 employees...")
    emp_df = generate_employees(n=80)
    emp_path = OUT_DIR / "employees.csv"
    emp_df.to_csv(emp_path, index=False)
    print(f"       Saved {len(emp_df)} employees → {emp_path}")

    print("\n[2/2] Generating 30 projects...")
    proj_df = generate_projects(n=30)
    proj_path = OUT_DIR / "projects.csv"
    proj_df.to_csv(proj_path, index=False)
    print(f"       Saved {len(proj_df)} projects → {proj_path}")

    # Quick sanity stats
    print("\n── Employee Stats ──────────────────────────────")
    print(emp_df["role"].value_counts().to_string())
    print(f"\nAvailability distribution:")
    print(emp_df["availability_pct"].value_counts().sort_index().to_string())

    print("\n── Project Stats ───────────────────────────────")
    print(f"Priority breakdown:")
    print(proj_df["priority"].value_counts().to_string())
    print(f"\nBudget bands:")
    print(proj_df["budget_band"].value_counts().to_string())
    print("\n All done!")


if __name__ == "__main__":
    main()