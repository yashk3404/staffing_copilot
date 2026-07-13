"""
Item 25 -- multi-user isolation for the wired-up matcher/optimizer
pipeline.

The v3 roadmap's item 25 exit criteria included a manual step: "a
manual run confirms a logged-in user's custom employee shows up as a
candidate in a custom-project solve, and does not leak into another
user's session." This file automates that instead of relying on a
one-off click-through.

Deliberately runs against tests/fake_supabase.py (via conftest.py's
autouse fixture + the opt-in second_user_supabase fixture), not a
real Supabase project -- test_rls_boundary.py (item 24) already
proves the Postgres RLS policies themselves hold at the database
level. This file proves the application code built on top of that
boundary (load_own_employees(), load_all_employees(),
get_busy_employee_ids(), staff_custom_project()) actually threads
owner scoping correctly end to end, through the real Matcher and
CP-SAT optimizer -- not just that storage returns the right rows.

Uses a real Matcher() (real embeddings, real 80-person CSV roster)
the same way test_optimizer.py's merged-pool test does, since the
thing under test is the *wiring*, not the store layer in isolation
(test_employee_store.py / test_project_store.py already cover that).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from matcher import Matcher
from optimize_staffing import staff_custom_project

from src.employee_store import save_employee, load_own_employees, load_all_employees
from src.project_store import save_project, get_candidate_pool

EMPTY_PLAN = pd.DataFrame(columns=["project_id", "role", "employee_id", "final_score"])

PLANT_NAME = "ZZZ Isolation Test Candidate"


@pytest.fixture(scope="module")
def matcher():
    """Loaded once for the whole file -- expensive (embeds the real
    80-person roster), and nothing here mutates the matcher itself."""
    m = Matcher()
    m.load()
    return m


def _real_roster(matcher):
    return matcher.emp_df.set_index("employee_id")


def _plant_employee():
    return save_employee({
        "name": PLANT_NAME,
        "role": "Backend Dev",
        "experience_years": 10,
        "availability_pct": 100,
        "skills": "Python",
        "department": "Engineering",
        "location": "Remote",
    })


def _plant_project():
    project_id = save_project({
        "project_name": "Isolation Test Project",
        "client": "Internal",
        "priority": "high",
        "min_experience": 2,
        "deadline_days": 30,
        "budget_band": "medium",
        "required_roles": "Backend Dev",
        "required_skills": "Python",
    })
    return {
        "project_id": project_id,
        "project_name": "Isolation Test Project",
        "min_experience": 2,
        "required_roles": "Backend Dev",
        "required_skills": "Python",
    }


class TestCustomEmployeeIsScoredForItsOwner:
    """Step 1-4 of the roadmap's manual walkthrough: User A's plant
    employee should be a real, scored candidate for User A's own
    project -- not merely present in storage."""

    def test_plant_employee_appears_in_own_employees_pool(self, matcher):
        _plant_employee()
        own = load_own_employees(_real_roster(matcher))
        assert PLANT_NAME in own["name"].values

    def test_plant_employee_scored_as_real_candidate_in_own_solve(self, matcher):
        _plant_employee()
        project = _plant_project()
        own = load_own_employees(_real_roster(matcher))

        staff_custom_project(
            project, matcher, EMPTY_PLAN, employees_df=own, verbose=False,
        )

        pool = get_candidate_pool(project["project_id"])
        assert pool is not None
        candidates = pool.get("Backend Dev")
        assert candidates is not None
        names = candidates["name"].tolist()
        assert PLANT_NAME in names, (
            "plant employee should be a scored candidate for their "
            "own role, whether or not they won the solve"
        )


class TestCustomEmployeeDoesNotLeakToAnotherUser:
    """Step 5-6 of the roadmap's manual walkthrough: log in as a
    second user and confirm User A's plant employee is invisible --
    not just that User B can't see User A's project."""

    def test_second_user_own_pool_excludes_first_users_plant(
        self, matcher, second_user_supabase,
    ):
        _plant_employee()  # User A

        second_user_supabase()  # switch to User B
        own_b = load_own_employees(_real_roster(matcher))
        assert PLANT_NAME not in own_b.get("name", pd.Series(dtype=object)).values

    def test_second_user_merged_pool_excludes_first_users_plant(
        self, matcher, second_user_supabase,
    ):
        _plant_employee()  # User A

        second_user_supabase()  # switch to User B
        merged_b = load_all_employees(_real_roster(matcher))
        assert PLANT_NAME not in merged_b["name"].values

    def test_second_users_solve_never_scores_first_users_plant(
        self, matcher, second_user_supabase,
    ):
        _plant_employee()  # User A, plant CE001

        second_user_supabase()  # switch to User B
        project_b = _plant_project()  # same role/skills, User B's own project
        own_b = load_own_employees(_real_roster(matcher))
        assert own_b.empty, (
            "sanity check: User B should have no employees of their "
            "own at this point in the test"
        )

        result = staff_custom_project(
            project_b, matcher, EMPTY_PLAN, employees_df=own_b, verbose=False,
        )
        assert result.empty, (
            "User B has zero eligible candidates for Backend Dev -- "
            "the solve must come back empty, not silently borrow "
            "User A's plant employee"
        )

        pool_b = get_candidate_pool(project_b["project_id"])
        if pool_b is not None:
            candidates = pool_b.get("Backend Dev")
            if candidates is not None and "name" in candidates.columns:
                assert PLANT_NAME not in candidates["name"].tolist()