"""
Shared pytest fixtures for the store-layer test suite.

Item 23 changed employee_store.py / project_store.py from
st.session_state-backed to Supabase-backed. Two things get faked here
so those modules can be imported and exercised headlessly:

1. A minimal `streamlit` stand-in (session_state dict access) -- a
   real Streamlit session only exists inside a running `streamlit run`
   process. Nothing else in this suite touches streamlit -- matcher.py
   and optimize_staffing.py don't import it -- so replacing it for the
   whole test session is safe.
2. A `FakeSupabaseClient` (tests/fake_supabase.py), monkeypatched over
   the `get_supabase_client` name each store module imported from
   src.auth, so no test ever makes a real network call or needs a live
   Supabase project.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from tests.fake_supabase import FakeSupabaseClient  # noqa: E402


class _FakeSessionState(dict):
    """Minimal stand-in for st.session_state: dict storage plus
    attribute-style access (st.session_state.custom_candidate_pools),
    which is exactly how the store modules use it for the one thing
    still session-scoped (save_candidate_pool/get_candidate_pool)."""
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
    # auth.py decorates get_supabase_client with @st.cache_resource at
    # import time -- a plain identity decorator is enough here since
    # tests never actually call the real function (get_supabase_client
    # itself is monkeypatched away below), they just need the module
    # to import cleanly.
    _fake_st.cache_resource = lambda fn=None, **kwargs: (fn if fn else (lambda f: f))
    sys.modules["streamlit"] = _fake_st

import streamlit as st  # noqa: E402 -- the fake module installed above


class _FakeUser:
    """Stand-in for the object Supabase Auth's sign_in_with_password()
    returns as result.user -- auth.py / the store modules only ever
    read .id and .email off of it."""
    def __init__(self, id="00000000-0000-0000-0000-000000000001",
                 email="test@example.com"):
        self.id = id
        self.email = email


@pytest.fixture(autouse=True)
def _reset_session_state():
    """Clears the fake session state before and after every test --
    each test starts from a blank slate for the candidate-pool cache
    (the one thing still session-scoped)."""
    st.session_state.clear()
    yield
    st.session_state.clear()


@pytest.fixture(autouse=True)
def _fake_supabase(monkeypatch, _reset_session_state):
    # Depending on _reset_session_state (rather than trusting autouse
    # ordering) guarantees this fixture's st.session_state["user"]
    # write happens AFTER that fixture's pre-test clear() -- autouse
    # fixtures with no explicit dependency are ordered alphabetically
    # by pytest, which otherwise runs this one FIRST and lets the
    # clear() wipe the login straight back out before the test body
    # ever runs.
    """
    Installs a fresh FakeSupabaseClient for every test and logs in a
    fake user, so employee_store.py / project_store.py run exactly as
    they would in the app, minus the network. autouse=True because
    every existing test in test_employee_store.py / test_project_store.py
    predates item 23 and doesn't know to request this explicitly --
    the fixture set (owner id CE001/C001 etc. sequencing) matches the
    old session-state tests' expectations as long as each test gets a
    single, consistent logged-in user and a wiped store, which this
    fixture guarantees.
    """
    import src.employee_store as employee_store
    import src.project_store as project_store

    client = FakeSupabaseClient()
    st.session_state["user"] = _FakeUser()

    monkeypatch.setattr(employee_store, "get_supabase_client", lambda: client)
    monkeypatch.setattr(project_store, "get_supabase_client", lambda: client)

    yield client


@pytest.fixture
def second_user_supabase(monkeypatch):
    """
    Opt-in fixture for tests that specifically want to prove two
    different users don't collide (item 23's 0002 migration fix).
    Swaps in a second fake user against the SAME underlying fake
    client the autouse fixture already installed, so both users'
    rows live side by side the way they would in one shared Postgres
    table.
    """
    import src.employee_store as employee_store
    import src.project_store as project_store

    def _switch_to_second_user():
        st.session_state["user"] = _FakeUser(
            id="00000000-0000-0000-0000-000000000002",
            email="second@example.com",
        )

    return _switch_to_second_user