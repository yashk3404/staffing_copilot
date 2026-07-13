"""
Unit tests for employee_store.py (Phase 3 item 6 / Phase 4-5 wiring).
Run: pytest tests/test_employee_store.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.employee_store import (
    save_employee, get_employee_by_id, load_all_employees,
    list_custom_employees, load_own_employees, delete_employee,
)


def _sample_record(name="Test Person", **overrides):
    record = {
        "name": name,
        "role": "Backend Dev",
        "experience_years": 4,
        "availability_pct": 80,
        "skills": "Python;Docker",
        "department": "Engineering",
        "location": "Pune",
    }
    record.update(overrides)
    return record


@pytest.fixture
def real_roster():
    """A tiny stand-in for employees_with_index.csv, indexed by
    employee_id the way dashboard.py loads it."""
    return pd.DataFrame([
        {"employee_id": "E001", "name": "Aryan Maharaj", "role": "Frontend Dev",
         "experience_years": 1, "availability_pct": 60, "skills": "JavaScript",
         "cost_band": "B", "department": "Analytics", "location": "Bangalore"},
        {"employee_id": "E002", "name": "Udant Dewan", "role": "Full Stack Dev",
         "experience_years": 10, "availability_pct": 60, "skills": "JavaScript;Node.js",
         "cost_band": "B", "department": "Engineering", "location": "Hyderabad"},
    ]).set_index("employee_id")


class TestSaveEmployee:
    def test_assigns_ce_id(self):
        emp_id = save_employee(_sample_record())
        assert emp_id == "CE001"

    def test_ids_increment_and_never_collide_with_real_range(self):
        id1 = save_employee(_sample_record("Person A"))
        id2 = save_employee(_sample_record("Person B"))
        assert id1 == "CE001"
        assert id2 == "CE002"
        assert not id1.startswith("E")

    def test_editing_in_place_does_not_create_duplicate(self):
        emp_id = save_employee(_sample_record("Person A"))
        record = get_employee_by_id(emp_id)
        record["role"] = "DevOps"
        second_id = save_employee({**record, "employee_id": emp_id})
        assert second_id == emp_id
        assert len(list_custom_employees()) == 1
        assert get_employee_by_id(emp_id)["role"] == "DevOps"


class TestGetEmployeeById:
    def test_finds_custom_employee_first(self):
        emp_id = save_employee(_sample_record())
        assert get_employee_by_id(emp_id)["name"] == "Test Person"

    def test_falls_back_to_real_roster(self, real_roster):
        result = get_employee_by_id("E001", employees_df=real_roster)
        assert result["name"] == "Aryan Maharaj"

    def test_returns_none_if_nowhere(self, real_roster):
        assert get_employee_by_id("E999", employees_df=real_roster) is None

    def test_returns_none_without_employees_df_for_unknown_id(self):
        assert get_employee_by_id("E001") is None


class TestLoadAllEmployees:
    def test_noop_with_no_custom_employees(self, real_roster):
        merged = load_all_employees(real_roster)
        assert len(merged) == len(real_roster)
        assert list(merged.index) == list(real_roster.index)

    def test_merges_custom_employee(self, real_roster):
        save_employee(_sample_record("Custom Dev"))
        merged = load_all_employees(real_roster)
        assert len(merged) == len(real_roster) + 1
        assert "CE001" in merged.index

    def test_missing_columns_filled_with_none_not_dropped(self, real_roster):
        # sample record has no "cost_band" -- real_roster does
        save_employee(_sample_record("Custom Dev"))
        merged = load_all_employees(real_roster)
        assert "cost_band" in merged.columns
        assert merged.loc["CE001", "cost_band"] is None

    def test_column_order_matches_real_roster(self, real_roster):
        save_employee(_sample_record("Custom Dev"))
        merged = load_all_employees(real_roster)
        assert list(merged.columns) == list(real_roster.columns)

    def test_does_not_mutate_original_roster(self, real_roster):
        original_cols = list(real_roster.columns)
        original_len = len(real_roster)
        save_employee(_sample_record("Custom Dev"))
        load_all_employees(real_roster)
        assert list(real_roster.columns) == original_cols
        assert len(real_roster) == original_len

    def test_dedupes_when_custom_employee_shares_an_id_with_the_real_roster(self, real_roster):
        # Regression test: scripts/migrate_demo_roster_to_user.py
        # intentionally saves custom employees using the SAME
        # employee_id as their real-roster row (e.g. "E001"). Without
        # deduping, .loc["E001"] would return a 2-row DataFrame
        # instead of a Series everywhere in the app.
        real_id = real_roster.index[0]
        record = _sample_record("Migrated Copy")
        record["employee_id"] = real_id
        save_employee(record)

        merged = load_all_employees(real_roster)
        assert len(merged) == len(real_roster)  # not +1
        assert not merged.index.duplicated().any()
        row = merged.loc[real_id]
        assert isinstance(row, pd.Series)
        # the custom/owned row wins over the original CSV row
        assert row["name"] == "Migrated Copy"


class TestListCustomEmployees:
    def test_empty_by_default(self):
        assert list_custom_employees() == []

    def test_returns_all_saved_records(self):
        save_employee(_sample_record("A"))
        save_employee(_sample_record("B"))
        names = {e["name"] for e in list_custom_employees()}
        assert names == {"A", "B"}


class TestLoadOwnEmployees:
    """v3 UI update -- custom projects match against ONLY the user's
    own saved employees, never the shared demo/synthetic roster."""

    def test_empty_when_no_custom_employees(self, real_roster):
        own = load_own_employees(real_roster)
        assert len(own) == 0
        # Same columns as the real roster, so it's a drop-in
        # employees_df= for match_adhoc()/staff_custom_project() even
        # when this user hasn't saved anyone yet.
        assert list(own.columns) == list(real_roster.columns)

    def test_excludes_demo_roster_rows(self, real_roster):
        save_employee(_sample_record("Custom Dev"))
        own = load_own_employees(real_roster)
        assert "E001" not in own.index
        assert "E002" not in own.index

    def test_includes_only_this_users_saved_employees(self, real_roster):
        save_employee(_sample_record("Custom Dev"))
        own = load_own_employees(real_roster)
        assert len(own) == 1
        assert "CE001" in own.index

    def test_does_not_mutate_original_roster(self, real_roster):
        original_len = len(real_roster)
        save_employee(_sample_record("Custom Dev"))
        load_own_employees(real_roster)
        assert len(real_roster) == original_len


class TestDeleteEmployee:
    def test_removes_the_employee(self):
        emp_id = save_employee(_sample_record("To Delete"))
        assert len(list_custom_employees()) == 1
        delete_employee(emp_id)
        assert list_custom_employees() == []

    def test_get_employee_by_id_returns_none_after_delete(self):
        emp_id = save_employee(_sample_record("To Delete"))
        delete_employee(emp_id)
        assert get_employee_by_id(emp_id) is None

    def test_does_not_affect_other_employees(self):
        keep_id = save_employee(_sample_record("Keep"))
        gone_id = save_employee(_sample_record("Gone"))
        delete_employee(gone_id)
        remaining = {e["employee_id"] for e in list_custom_employees()}
        assert remaining == {keep_id}

    def test_noop_for_unknown_id(self):
        delete_employee("CE999")  # must not raise