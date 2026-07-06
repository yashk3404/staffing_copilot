# tests/test_data_generator.py
"""
Unit tests for synthetic data generation.
Run: pytest tests/test_data_generator.py -v
"""

import pytest
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def test_employees_file_exists():
    assert (DATA_DIR / "employees_with_index.csv").exists()


def test_projects_file_exists():
    assert (DATA_DIR / "projects_with_index.csv").exists()


def test_employee_count():
    df = pd.read_csv(DATA_DIR / "employees_with_index.csv")
    assert len(df) == 80, f"Expected 80 employees, got {len(df)}"


def test_project_count():
    df = pd.read_csv(DATA_DIR / "projects_with_index.csv")
    assert len(df) == 30, f"Expected 30 projects, got {len(df)}"


def test_employee_columns():
    df = pd.read_csv(DATA_DIR / "employees_with_index.csv")
    required = ["employee_id", "name", "role", "experience_years",
                "skills", "availability_pct"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_project_columns():
    df = pd.read_csv(DATA_DIR / "projects_with_index.csv")
    required = ["project_id", "project_name", "client",
                "required_roles", "required_skills", "min_experience"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_employee_ids_unique():
    df = pd.read_csv(DATA_DIR / "employees_with_index.csv")
    assert df["employee_id"].nunique() == len(df), \
        "Duplicate employee IDs found"


def test_project_ids_unique():
    df = pd.read_csv(DATA_DIR / "projects_with_index.csv")
    assert df["project_id"].nunique() == len(df), \
        "Duplicate project IDs found"


def test_availability_range():
    df = pd.read_csv(DATA_DIR / "employees_with_index.csv")
    assert df["availability_pct"].between(0, 100).all(), \
        "availability_pct out of 0–100 range"


def test_experience_positive():
    df = pd.read_csv(DATA_DIR / "employees_with_index.csv")
    assert (df["experience_years"] > 0).all(), \
        "Found employee with 0 or negative experience"


def test_no_null_employee_names():
    df = pd.read_csv(DATA_DIR / "employees_with_index.csv")
    assert df["name"].notna().all(), "Null names in employees"


def test_no_null_project_names():
    df = pd.read_csv(DATA_DIR / "projects_with_index.csv")
    assert df["project_name"].notna().all(), "Null names in projects"