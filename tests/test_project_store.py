"""
Unit tests for project_store.py (Phase 3 item 9 / Phase 3-4 wiring).
Run: pytest tests/test_project_store.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.project_store import (
    save_project, update_project_assignments, get_project_by_id,
    load_all_projects, list_custom_projects, get_busy_employee_ids,
    get_capacity_summary,
)
from src.employee_store import save_employee


def _sample_project(name="Test Project", **overrides):
    record = {
        "project_name": name,
        "client": "Internal",
        "priority": "high",
        "min_experience": 2,
        "deadline_days": 30,
        "budget_band": "medium",
        "required_roles": "Backend Dev;Frontend Dev",
        "required_skills": "Python;React",
    }
    record.update(overrides)
    return record


@pytest.fixture
def real_projects():
    return pd.DataFrame([
        {"project_id": "P001", "project_name": "Alpha", "client": "Acme",
         "priority": "high", "min_experience": 2, "deadline_days": 30,
         "budget_band": "medium", "required_roles": "Backend Dev",
         "required_skills": "Python"},
    ]).set_index("project_id")


@pytest.fixture
def real_employees():
    return pd.DataFrame([
        {"employee_id": "E001", "name": "A", "role": "Backend Dev",
         "experience_years": 3, "availability_pct": 80, "skills": "Python"},
        {"employee_id": "E002", "name": "B", "role": "Frontend Dev",
         "experience_years": 5, "availability_pct": 50, "skills": "React"},
        {"employee_id": "E003", "name": "C", "role": "Backend Dev",
         "experience_years": 1, "availability_pct": 90, "skills": "Python"},
    ]).set_index("employee_id")


@pytest.fixture
def empty_plan():
    return pd.DataFrame(columns=["project_id", "role", "employee_id", "final_score"])


class TestSaveProject:
    def test_assigns_c_id(self):
        pid = save_project(_sample_project())
        assert pid == "C001"

    def test_defaults_assignments_to_empty_dict(self):
        pid = save_project(_sample_project())
        assert get_project_by_id(pid)["assignments"] == {}

    def test_ids_never_collide_with_real_range(self):
        pid = save_project(_sample_project())
        assert not pid.startswith("P")


class TestUpdateProjectAssignments:
    def test_writes_assignments(self):
        pid = save_project(_sample_project())
        update_project_assignments(pid, {"Backend Dev": "E001"})
        assert get_project_by_id(pid)["assignments"] == {"Backend Dev": "E001"}

    def test_noop_for_unknown_project(self):
        update_project_assignments("C999", {"Backend Dev": "E001"})  # must not raise


class TestGetProjectById:
    def test_finds_custom_project_first(self):
        pid = save_project(_sample_project())
        assert get_project_by_id(pid)["project_name"] == "Test Project"

    def test_falls_back_to_real_projects(self, real_projects):
        assert get_project_by_id("P001", projects_df=real_projects)["project_name"] == "Alpha"

    def test_returns_none_if_nowhere(self, real_projects):
        assert get_project_by_id("P999", projects_df=real_projects) is None


class TestLoadAllProjects:
    def test_noop_with_no_custom_projects(self, real_projects):
        merged = load_all_projects(real_projects)
        assert len(merged) == len(real_projects)

    def test_merges_and_drops_assignments_column(self, real_projects):
        pid = save_project(_sample_project())
        update_project_assignments(pid, {"Backend Dev": "E001"})
        merged = load_all_projects(real_projects)
        assert pid in merged.index
        assert "assignments" not in merged.columns


class TestGetBusyEmployeeIds:
    def test_includes_premade_plan_assignments(self):
        plan = pd.DataFrame([
            {"project_id": "P001", "role": "Backend Dev",
             "employee_id": "E001", "final_score": 0.9},
        ])
        assert get_busy_employee_ids(plan) == {"E001"}

    def test_includes_custom_project_assignments(self, empty_plan):
        pid = save_project(_sample_project())
        update_project_assignments(pid, {"Backend Dev": "E002", "Frontend Dev": "CE001"})
        assert get_busy_employee_ids(empty_plan) == {"E002", "CE001"}

    def test_unstaffed_projects_contribute_nothing(self, empty_plan):
        save_project(_sample_project())  # never staffed
        assert get_busy_employee_ids(empty_plan) == set()

    def test_combines_premade_and_custom(self):
        plan = pd.DataFrame([
            {"project_id": "P001", "role": "Backend Dev",
             "employee_id": "E001", "final_score": 0.9},
        ])
        pid = save_project(_sample_project())
        update_project_assignments(pid, {"Backend Dev": "E002"})
        assert get_busy_employee_ids(plan) == {"E001", "E002"}


class TestGetCapacitySummary:
    def test_total_pool_includes_custom_employees(self, real_employees, empty_plan):
        save_employee({"name": "X", "role": "Backend Dev", "experience_years": 5,
                        "availability_pct": 100, "skills": "Python",
                        "department": "Eng", "location": "Remote"})
        project = _sample_project(required_roles="Backend Dev")
        summary = get_capacity_summary(project, real_employees, empty_plan)
        assert summary["total_pool"] == len(real_employees) + 1

    def test_busy_employees_excluded_from_available(self, real_employees):
        plan = pd.DataFrame([
            {"project_id": "P001", "role": "Backend Dev",
             "employee_id": "E001", "final_score": 0.9},
        ])
        project = _sample_project(required_roles="Backend Dev")
        summary = get_capacity_summary(project, real_employees, plan)
        # E001 is busy; E003 is Backend Dev + available -> only E003 counted
        assert summary["by_role"]["Backend Dev"]["available"] == 1

    def test_low_availability_excluded(self, real_employees, empty_plan):
        # E002 has availability_pct 50, below default min_avail=60
        project = _sample_project(required_roles="Frontend Dev")
        summary = get_capacity_summary(project, real_employees, empty_plan)
        assert summary["by_role"]["Frontend Dev"]["in_role"] == 1
        assert summary["by_role"]["Frontend Dev"]["available"] == 0

    def test_does_not_double_count_capacity_summary_already_merges(self, real_employees, empty_plan):
        # get_capacity_summary() calls load_all_employees() internally --
        # passing it an ALREADY-merged frame would double-concat custom
        # employees. This locks in the single-merge contract.
        save_employee({"name": "X", "role": "Backend Dev", "experience_years": 5,
                        "availability_pct": 100, "skills": "Python",
                        "department": "Eng", "location": "Remote"})
        project = _sample_project(required_roles="Backend Dev")
        summary = get_capacity_summary(project, real_employees, empty_plan)
        assert summary["total_pool"] == len(real_employees) + 1  # not +2