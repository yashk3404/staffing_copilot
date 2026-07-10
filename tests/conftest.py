"""
Shared pytest fixtures for the Phase 5 test suite.

employee_store.py and project_store.py are built against
st.session_state -- a real Streamlit session only exists inside a
running `streamlit run` process, so a lightweight, dict-backed
stand-in is installed here (module level, before any test file gets
collected) so those two modules can be imported and exercised
headless. Nothing else in this suite touches streamlit -- matcher.py
and optimize_staffing.py don't import it -- so replacing it for the
whole test session is safe.
"""
import sys
import types

import pytest


class _FakeSessionState(dict):
    """
    Minimal stand-in for st.session_state: dict storage plus
    attribute-style access (st.session_state.custom_employees), which
    is exactly how employee_store.py / project_store.py use it.
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


if "streamlit" not in sys.modules or not hasattr(sys.modules["streamlit"], "_is_fake_for_tests"):
    _fake_st = types.ModuleType("streamlit")
    _fake_st._is_fake_for_tests = True
    _fake_st.session_state = _FakeSessionState()
    sys.modules["streamlit"] = _fake_st

import streamlit as st  # noqa: E402 -- the fake module installed above


@pytest.fixture(autouse=True)
def _reset_session_state():
    """
    employee_store.py / project_store.py both key off
    st.session_state.custom_employees / custom_projects (and their
    counters). Without a reset, an employee added in one test would
    still be there in the next -- this clears the fake session state
    before and after every test so each one starts from a blank slate.
    """
    st.session_state.clear()
    yield
    st.session_state.clear()