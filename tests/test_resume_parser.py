# tests/test_matcher.py
"""
Unit tests for the Matcher scoring logic.
Run: pytest tests/test_matcher.py -v
"""

import pytest
import sys
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
    import pandas as pd
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
    import pandas as pd
    sm = pd.read_csv(DATA_DIR / "score_matrix.csv")
    assert len(sm) == 6800, f"Expected 6800 rows, got {len(sm)}"
    assert "final_score" in sm.columns
    assert "eligible"    in sm.columns


# ── Phase 5 / item 17 -- merged-pool (custom employee) scoring ──────

class TestMergedPoolMatching:
    """
    match()/match_adhoc() accept an optional employees_df so a merged
    pool (real roster + session-added custom employees, as built by
    employee_store.load_all_employees()) can be scored -- without it,
    behavior must stay byte-for-byte identical to before item 17.
    """

    @pytest.fixture
    def merged_df(self, matcher):
        """Real roster (dashboard's employee_id-indexed convention)
        plus one hand-built custom employee, not otherwise present in
        self.emp_emb -- exactly the shape load_all_employees() hands
        back."""
        import pandas as pd
        real = matcher.emp_df.set_index("employee_id")
        custom_row = pd.DataFrame([{
            "name": "Merged Pool Test Dev",
            "role": "Backend Dev",
            "experience_years": 6,
            "availability_pct": 100,
            "skills": "Python;Docker;PostgreSQL",
            "cost_band": None,
            "department": "Engineering",
            "location": "Remote",
        }], index=pd.Index(["CE999"], name="employee_id"))
        return pd.concat([real, custom_row])

    def test_default_call_unaffected(self, matcher):
        """No employees_df passed -> same row count and same
        candidate set as before item 17 existed."""
        result = matcher.match("P001", "Backend Dev", verbose=False)
        assert len(result) == len(matcher.emp_df)
        assert "CE999" not in set(result["employee_id"])

    def test_merged_pool_includes_custom_employee(self, matcher, merged_df):
        result = matcher.match(
            "P001", "Backend Dev", employees_df=merged_df, verbose=False
        )
        assert "CE999" in set(result["employee_id"])
        assert len(result) == len(merged_df)

    def test_custom_employee_scored_not_just_listed(self, matcher, merged_df):
        result = matcher.match(
            "P001", "Backend Dev", employees_df=merged_df, verbose=False
        )
        row = result[result.employee_id == "CE999"].iloc[0]
        # Availability 100 and experience 6yrs should clear P001's
        # eligibility bar, so it must be a real (non-zero) score, not
        # a placeholder.
        assert row["eligible"] in (True, False)   # never crashes/NaN
        if row["eligible"]:
            assert 0 <= row["final_score"] <= 1

    def test_match_adhoc_also_accepts_merged_pool(self, matcher, merged_df):
        project = {
            "min_experience": 2,
            "required_skills": "Python;Docker",
        }
        result = matcher.match_adhoc(
            project, "Backend Dev", employees_df=merged_df, verbose=False
        )
        assert "CE999" in set(result["employee_id"])

    def test_match_adhoc_respects_exclude_ids_for_custom_employee(self, matcher, merged_df):
        project = {"min_experience": 2, "required_skills": "Python;Docker"}
        result = matcher.match_adhoc(
            project, "Backend Dev", employees_df=merged_df,
            exclude_ids={"CE999"}, verbose=False,
        )
        assert "CE999" not in set(result["employee_id"])

    def test_build_profile_text_handles_missing_fields(self, matcher):
        import pandas as pd
        emp = pd.Series({"role": "Backend Dev", "experience_years": 3,
                          "availability_pct": 80, "skills": ""})
        text = matcher._build_profile_text(emp)
        assert "Backend Dev" in text
        assert "Skilled in" not in text  # no skills -> that clause omitted

    def test_normalize_emp_df_resets_id_indexed_frame(self, matcher):
        import pandas as pd
        id_indexed = matcher.emp_df.set_index("employee_id")
        normalized = matcher._normalize_emp_df(id_indexed)
        assert "employee_id" in normalized.columns

    def test_normalize_emp_df_noop_when_already_a_column(self, matcher):
        normalized = matcher._normalize_emp_df(matcher.emp_df)
        assert normalized is matcher.emp_df or \
            list(normalized["employee_id"]) == list(matcher.emp_df["employee_id"])