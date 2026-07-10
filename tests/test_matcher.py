# tests/test_matcher.py
"""
Unit tests for the Matcher scoring logic.
Run: pytest tests/test_matcher.py -v
"""

import pytest
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from matcher import Matcher

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


@pytest.fixture(scope="module")
def matcher():
    m = Matcher()
    m.load()
    return m


def test_matcher_loads(matcher):
    assert matcher.emp_df   is not None
    assert matcher.proj_df  is not None
    assert matcher.emp_emb  is not None
    assert len(matcher.emp_df)  == 80
    assert len(matcher.proj_df) == 30


def test_match_returns_dataframe(matcher):
    result = matcher.match("P001", "Backend Dev", verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 80   # one row per employee


def test_match_required_columns(matcher):
    result = matcher.match("P001", "Backend Dev", verbose=False)
    required = ["employee_id", "final_score", "eligible",
                "semantic_score", "skill_overlap"]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_match_scores_in_range(matcher):
    result = matcher.match("P001", "Backend Dev", verbose=False)
    eligible = result[result.eligible == True]
    assert (eligible["final_score"] >= 0).all()
    assert (eligible["final_score"] <= 1).all()


def test_ineligible_score_zero(matcher):
    result = matcher.match("P001", "Backend Dev", verbose=False)
    ineligible = result[result.eligible == False]
    assert (ineligible["final_score"] == 0.0).all(), \
        "Ineligible employees should have score 0"


def test_sorted_descending(matcher):
    result = matcher.match("P001", "Backend Dev", verbose=False)
    scores = result["final_score"].tolist()
    assert scores == sorted(scores, reverse=True), \
        "Results not sorted descending by final_score"


def test_availability_factor_logic(matcher):
    # Below min → 0.0
    assert matcher._availability_factor(50, min_avail=60) == 0.0
    # At min → 0.0
    assert matcher._availability_factor(60, min_avail=60) == 0.0
    # At max → 1.0
    assert matcher._availability_factor(100, min_avail=60) == 1.0


def test_experience_factor_logic(matcher):
    # Below min → 0.0
    assert matcher._experience_factor(1, min_years=4) == 0.0
    # At min → 0.5
    assert matcher._experience_factor(4, min_years=4) == 0.5


def test_skill_overlap_logic(matcher):
    # All skills match
    assert matcher._skill_overlap("python;sql;aws",
                                   ["python", "sql", "aws"]) == 1.0
    # No skills match
    assert matcher._skill_overlap("react;css",
                                   ["python", "sql"]) == 0.0
    # Partial match
    overlap = matcher._skill_overlap("python;react",
                                      ["python", "sql"])
    assert overlap == 0.5


def test_invalid_project_raises(matcher):
    with pytest.raises(ValueError):
        matcher.match("PXXX", "Backend Dev", verbose=False)


def test_score_matrix_file_exists():
    sm = pd.read_csv(DATA_DIR / "score_matrix.csv")
    assert len(sm) == 6800, f"Expected 6800 rows, got {len(sm)}"
    assert "final_score" in sm.columns
    assert "eligible"    in sm.columns


# ── Phase 5 / Item 19: Custom Employee Pipeline Verification ─────────

def test_dynamic_custom_employee_embedding(matcher):
    """
    Test that injecting a custom employee (not in the precomputed embeddings)
    successfully generates an on-the-fly embedding and returns a valid match score.
    """
    # Create a mock custom employee dataframe
    custom_emp = pd.DataFrame([{
        "employee_id": "CE999",
        "name": "Test Custom Employee",
        "role": "Backend Dev",
        "experience_years": 8,
        "availability_pct": 100,
        "cost_band": "medium",
        "skills": "Python;FastAPI;PostgreSQL;Docker",
        "department": "Engineering",
        "location": "Remote"
    }])
    
    # Concat with a slice of the real roster to simulate a merged pool
    merged_pool = pd.concat([matcher.emp_df.head(5), custom_emp], ignore_index=True)
    
    # Run the match for a Backend Dev role
    result = matcher.match(
        project_id="P001", 
        role="Backend Dev", 
        top_k=10, 
        verbose=False, 
        employees_df=merged_pool
    )
    
    # Assertions
    assert not result.empty, "Match result should not be empty"
    assert "CE999" in result["employee_id"].values, "Custom employee was dropped from the candidate pool"
    
    # Check that the custom employee received a valid score
    ce_result = result[result["employee_id"] == "CE999"].iloc[0]
    assert 0.0 <= ce_result["semantic_score"] <= 1.0, "Semantic score out of bounds"
    assert 0.0 <= ce_result["final_score"] <= 1.0, "Final score out of bounds"
    assert ce_result["eligible"] == True, "Custom employee should be marked eligible based on stats"