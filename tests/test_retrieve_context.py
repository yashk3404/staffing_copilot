"""
Unit tests for retrieve_context.py -- specifically item 22's
retrieve_adhoc(), which exists to be a drop-in replacement for
retrieve() when explaining a custom (C0xx) project's assignment
instead of a premade one.

The tests here don't assert on retrieve()'s OR retrieve_adhoc()'s
*values* -- generate_explanation.build_prompt() and dashboard.py's
runner-up panel don't care what the numbers are, they care that both
functions hand back a dict with the same keys, nested the same way.
A silent schema drift between the two paths (e.g. retrieve_adhoc()
renaming a key, or nesting "score" one level differently) would only
surface as a KeyError deep inside whichever caller happens to run
first against a custom project -- these tests catch that at the
source instead.

Run: pytest tests/test_retrieve_context.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from retrieve_context import ContextRetriever, build_project_summary

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# retrieve()'s expected key shape, one level at a time -- asserted
# against explicitly (not just "same keys as some other call") so a
# key silently disappearing from BOTH functions at once (e.g. an
# unrelated refactor) still gets caught, not just drift between them.
TOP_LEVEL_KEYS   = {"project_id", "role", "project", "assigned", "runner_up", "n_eligible"}
PROJECT_KEYS     = {"project_id", "name", "client", "deadline", "skills", "summary"}
ASSIGNED_KEYS    = {"employee_id", "name", "actual_role", "experience_years",
                     "availability_pct", "skills", "score", "profile"}
RUNNER_UP_KEYS   = {"employee_id", "name", "actual_role", "experience_years",
                     "availability_pct", "skills", "score", "score_gap"}


@pytest.fixture(scope="module")
def retriever():
    # ContextRetriever only reads CSV/JSON in __init__ -- no
    # sentence_transformers import anywhere in retrieve_context.py --
    # so this works in any sandbox, unlike matcher.py-dependent tests.
    return ContextRetriever(str(DATA_DIR))


@pytest.fixture
def hand_built_pool():
    """Same shape match_all_roles_adhoc() returns: {role: DataFrame}."""
    return {
        "Backend Dev": pd.DataFrame([
            {"employee_id": "E900", "name": "Chosen Dev", "role": "Backend Dev",
             "experience_years": 5, "availability_pct": 80, "skills": "Python;SQL",
             "final_score": 0.75, "eligible": True},
            {"employee_id": "E901", "name": "Runner Up Dev", "role": "Backend Dev",
             "experience_years": 4, "availability_pct": 70, "skills": "Python",
             "final_score": 0.60, "eligible": True},
        ]),
    }


@pytest.fixture
def hand_built_project():
    return {
        "project_id": "C900",
        "project_name": "Schema Parity Test Project",
        "client": "Test Client",
        "required_roles": "Backend Dev",
        "required_skills": "Python; SQL",
        "min_experience": 2,
        "deadline_days": 30,
        "priority": "High",
        "budget_band": "Medium",
        "assignments": {"Backend Dev": "E900"},
    }


class TestSchemaParity:
    def test_top_level_keys_match(self, retriever, hand_built_project, hand_built_pool):
        real  = retriever.retrieve("P001", "Backend Dev")
        adhoc = retriever.retrieve_adhoc(
            "C900", "Backend Dev", hand_built_project, hand_built_pool
        )
        assert set(real.keys())  == TOP_LEVEL_KEYS
        assert set(adhoc.keys()) == TOP_LEVEL_KEYS

    def test_project_dict_keys_match(self, retriever, hand_built_project, hand_built_pool):
        real  = retriever.retrieve("P001", "Backend Dev")
        adhoc = retriever.retrieve_adhoc(
            "C900", "Backend Dev", hand_built_project, hand_built_pool
        )
        assert set(real["project"].keys())  == PROJECT_KEYS
        assert set(adhoc["project"].keys()) == PROJECT_KEYS

    def test_assigned_dict_keys_match(self, retriever, hand_built_project, hand_built_pool):
        real  = retriever.retrieve("P001", "Backend Dev")
        adhoc = retriever.retrieve_adhoc(
            "C900", "Backend Dev", hand_built_project, hand_built_pool
        )
        assert set(real["assigned"].keys())  == ASSIGNED_KEYS
        assert set(adhoc["assigned"].keys()) == ASSIGNED_KEYS

    def test_runner_up_dict_keys_match_when_present(
        self, retriever, hand_built_project, hand_built_pool
    ):
        real  = retriever.retrieve("P001", "Backend Dev")
        adhoc = retriever.retrieve_adhoc(
            "C900", "Backend Dev", hand_built_project, hand_built_pool
        )
        assert real["runner_up"]  is not None, \
            "P001/Backend Dev is expected to have a runner-up in the bundled data"
        assert adhoc["runner_up"] is not None, \
            "hand_built_pool was deliberately given 2 candidates"
        assert set(real["runner_up"].keys())  == RUNNER_UP_KEYS
        assert set(adhoc["runner_up"].keys()) == RUNNER_UP_KEYS

    def test_runner_up_is_none_not_missing_key_when_absent(
        self, retriever, hand_built_project
    ):
        """Single-candidate role-slot -- runner_up must be present as
        the key with value None, not simply absent from the dict (a
        caller doing ctx.get("runner_up") vs ctx["runner_up"] would
        behave differently, and generate_explanation.py uses the
        latter)."""
        single_pool = {"Backend Dev": pd.DataFrame([
            {"employee_id": "E900", "name": "Only Candidate", "role": "Backend Dev",
             "experience_years": 5, "availability_pct": 80, "skills": "Python",
             "final_score": 0.75, "eligible": True},
        ])}
        adhoc = retriever.retrieve_adhoc(
            "C900", "Backend Dev", hand_built_project, single_pool
        )
        assert "runner_up" in adhoc
        assert adhoc["runner_up"] is None

    def test_error_shape_matches(self, retriever, hand_built_project, hand_built_pool):
        """Both functions signal failure the same way: a dict with
        exactly one key, "error" -- not an exception, not a
        differently-shaped dict."""
        real_err  = retriever.retrieve("P001", "Role That Does Not Exist")
        adhoc_err = retriever.retrieve_adhoc(
            "C900", "Role That Does Not Exist", hand_built_project, hand_built_pool
        )
        assert set(real_err.keys())  == {"error"}
        assert set(adhoc_err.keys()) == {"error"}

    def test_adhoc_missing_pool_error_shape(self, retriever, hand_built_project):
        adhoc_err = retriever.retrieve_adhoc(
            "C900", "Backend Dev", hand_built_project, None
        )
        assert set(adhoc_err.keys()) == {"error"}

    def test_score_fields_are_floats_in_both(
        self, retriever, hand_built_project, hand_built_pool
    ):
        """generate_explanation.py formats ctx['assigned']['score'] --
        a str or numpy type surviving here would still work today but
        is a schema-drift risk (e.g. json-serializing the ctx dict for
        caching/logging later)."""
        real  = retriever.retrieve("P001", "Backend Dev")
        adhoc = retriever.retrieve_adhoc(
            "C900", "Backend Dev", hand_built_project, hand_built_pool
        )
        assert isinstance(real["assigned"]["score"],  float)
        assert isinstance(adhoc["assigned"]["score"], float)


class TestBuildProjectSummary:
    """build_project_summary() is retrieve_adhoc()'s on-the-fly
    stand-in for the offline project_profiles.json 'summary' field --
    not asserting exact wording (that's a display detail), just that
    it degrades gracefully rather than crashing on sparse records."""

    def test_full_record_produces_nonempty_summary(self, hand_built_project):
        assert len(build_project_summary(hand_built_project)) > 0

    def test_sparse_record_does_not_crash(self):
        summary = build_project_summary({"project_name": "Bare"})  # must not raise
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_completely_empty_record_does_not_crash(self):
        summary = build_project_summary({})  # must not raise
        assert isinstance(summary, str)