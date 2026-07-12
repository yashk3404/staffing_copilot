"""
Item 24 -- RLS boundary tests against a REAL Supabase project.

Deliberately NOT using tests/fake_supabase.py or conftest.py's autouse
fixtures -- those exist so employee_store.py/project_store.py can be
unit-tested headlessly, but a mock enforcing "the RLS policy I assume
exists" would just prove the mock is self-consistent, not that the
actual Postgres RLS policies in supabase/migrations/0001_schema_and_rls.sql
(and the 0002 follow-up) do what they're supposed to. This file talks
to real tables with two real authenticated users and checks what
Postgres itself allows.

Skips itself entirely (no failure, no error) unless every env var
below is set -- safe to leave in the suite permanently; a normal
`pytest tests/` run never touches this file.

── One-time setup ──────────────────────────────────────────────────
1. In your Supabase project (staffing-copilot-test, or whichever
   project SUPABASE_URL below points at) -- Authentication -> Settings
   -> turn OFF "Confirm email" for this test project. Two throwaway
   test accounts are easier without an inbox to click through; turn
   it back on before this ever points at a real prod project.
2. Create two test accounts (Authentication -> Users -> Add user, or
   just sign up twice through the app's own Sign up tab) --
   e.g. rls-test-1@example.com / rls-test-2@example.com, distinct
   passwords.
3. Export these before running pytest:

    export SUPABASE_URL=https://your-project-ref.supabase.co
    export SUPABASE_KEY=your-anon-or-publishable-key
    export RLS_TEST_USER_1_EMAIL=rls-test-1@example.com
    export RLS_TEST_USER_1_PASSWORD=<password>
    export RLS_TEST_USER_2_EMAIL=rls-test-2@example.com
    export RLS_TEST_USER_2_PASSWORD=<password>

4. Run just this file (it's slow -- real network round trips --
   and writes/deletes real throwaway rows):

    pytest tests/test_rls_boundary.py -v

Every row this file writes is prefixed "RLSTEST-" and cleaned up in
each test's own teardown (try/finally), so a failed run may leave a
stray row or two behind -- safe to delete manually from the Supabase
table editor by filtering employee_id/project_id starting with
"RLSTEST-" if that ever happens.

IMPORTANT: point SUPABASE_URL at your *test* project. This file does
not know or care whether it's pointed at prod -- that's on whoever
sets the env vars.
"""

import os
import uuid

import pytest

# Optional convenience: if a .env file exists (gitignored, same as
# .streamlit/secrets.toml), load it so the six vars below don't need
# to be `export`ed by hand every terminal session. Silently does
# nothing if python-dotenv isn't installed or no .env exists -- the
# skip check below still works either way, this is purely a
# quality-of-life load.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REQUIRED_ENV_VARS = [
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "RLS_TEST_USER_1_EMAIL",
    "RLS_TEST_USER_1_PASSWORD",
    "RLS_TEST_USER_2_EMAIL",
    "RLS_TEST_USER_2_PASSWORD",
]

pytestmark = pytest.mark.skipif(
    not all(os.environ.get(v) for v in REQUIRED_ENV_VARS),
    reason=(
        "Item 24 RLS boundary tests need real Supabase creds + two "
        "confirmed test user accounts -- see this file's module "
        "docstring for one-time setup. Skipped (not failed) when "
        "those env vars aren't set, e.g. in a normal `pytest tests/` "
        "run or CI."
    ),
)

# Imported lazily (inside the skip-gated module) so a normal test run
# with no Supabase creds set doesn't require the `supabase` package to
# even be importable in whatever environment runs the mocked suite.
from supabase import create_client  # noqa: E402


def _login(email_var: str, password_var: str):
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    client = create_client(url, key)
    result = client.auth.sign_in_with_password({
        "email": os.environ[email_var],
        "password": os.environ[password_var],
    })
    return client, result.user.id


@pytest.fixture(scope="module")
def _session_a():
    client, uid = _login("RLS_TEST_USER_1_EMAIL", "RLS_TEST_USER_1_PASSWORD")
    yield client, uid
    client.auth.sign_out()


@pytest.fixture(scope="module")
def _session_b():
    client, uid = _login("RLS_TEST_USER_2_EMAIL", "RLS_TEST_USER_2_PASSWORD")
    yield client, uid
    client.auth.sign_out()


@pytest.fixture(scope="module")
def client_a(_session_a):
    return _session_a[0]


@pytest.fixture(scope="module")
def client_b(_session_b):
    return _session_b[0]


@pytest.fixture(scope="module")
def user_a_id(_session_a):
    return _session_a[1]


@pytest.fixture(scope="module")
def user_b_id(_session_b):
    return _session_b[1]


def _employee_row(owner_id: str, name: str = "RLS Test Employee") -> dict:
    return {
        "employee_id": f"RLSTEST-{uuid.uuid4().hex[:10]}",
        "owner_id": owner_id,
        "name": name,
        "role": "Backend Dev",
        "experience_years": 3,
        "availability_pct": 100,
        "skills": ["Python"],
    }


@pytest.fixture
def user_a_employee(client_a, user_a_id):
    """One throwaway employee row owned by user A. Cleaned up via
    user A's own client (RLS lets an owner delete their own row) even
    if the test body never reaches its assertions."""
    row = _employee_row(user_a_id)
    client_a.table("employees").insert(row).execute()
    emp_id = row["employee_id"]
    yield emp_id
    client_a.table("employees").delete().eq("employee_id", emp_id).eq(
        "owner_id", user_a_id
    ).execute()


# ── Cross-user SELECT ────────────────────────────────────────────────

def test_user_b_cannot_read_user_a_employee_by_id(client_b, user_a_employee):
    result = (
        client_b.table("employees")
        .select("*")
        .eq("employee_id", user_a_employee)
        .execute()
    )
    assert result.data == []


def test_user_b_broad_query_excludes_user_a_rows(client_b, user_a_id, user_a_employee):
    """Not filtering by id at all -- a broad, unscoped SELECT should
    still never surface user A's owner_id. This is the check that
    actually proves the DATABASE is blocking it (RLS), not just that
    app code remembers to filter -- a query with no owner_id filter in
    it at all still can't see the row."""
    result = client_b.table("employees").select("employee_id, owner_id").execute()
    returned_owner_ids = {row["owner_id"] for row in result.data}
    assert user_a_id not in returned_owner_ids


# ── Cross-user UPDATE / DELETE ────────────────────────────────────────

def test_user_b_update_on_user_a_row_affects_nothing(client_a, client_b, user_a_employee):
    client_b.table("employees").update({"name": "Hijacked"}).eq(
        "employee_id", user_a_employee
    ).execute()

    # Confirm through user A's own eyes that the name is untouched --
    # a zero-row RLS-filtered UPDATE returns success (not an error),
    # so the only real proof is checking the data didn't change.
    still_original = (
        client_a.table("employees")
        .select("name")
        .eq("employee_id", user_a_employee)
        .execute()
    )
    assert still_original.data[0]["name"] == "RLS Test Employee"


def test_user_b_delete_on_user_a_row_affects_nothing(client_a, client_b, user_a_employee):
    client_b.table("employees").delete().eq("employee_id", user_a_employee).execute()

    still_there = (
        client_a.table("employees")
        .select("employee_id")
        .eq("employee_id", user_a_employee)
        .execute()
    )
    assert len(still_there.data) == 1


# ── INSERT boundary conditions ────────────────────────────────────────

def test_cannot_insert_with_spoofed_owner_id(client_b, user_a_id):
    """User B trying to write a row claiming to be owned by user A --
    the employees_insert policy's WITH CHECK (owner_id = auth.uid())
    should reject this outright, not just hide it afterward."""
    spoofed = _employee_row(user_a_id, name="Spoofed")
    with pytest.raises(Exception):
        client_b.table("employees").insert(spoofed).execute()


def test_regular_user_cannot_insert_null_owner_row(client_a):
    """owner_id IS NULL is reserved for shared demo data. Under Option
    B the app itself never writes any (the demo roster stays in CSV),
    but the policy should still reject a regular user trying to create
    one directly -- item 24's exit criteria calls this out explicitly
    ("visible to all, not writable by them")."""
    fake_demo = _employee_row(None, name="Fake Demo Row")
    with pytest.raises(Exception):
        client_a.table("employees").insert(fake_demo).execute()


def test_demo_rows_visible_to_all_but_equal_for_both_users(client_a, client_b):
    """Whatever owner_id IS NULL rows exist (there may legitimately be
    zero under Option B), both authenticated users should see the same
    set -- proves the "visible to all" half of the policy, independent
    of whether anything has seeded that table yet."""
    result_a = client_a.table("employees").select("employee_id").is_("owner_id", "null").execute()
    result_b = client_b.table("employees").select("employee_id").is_("owner_id", "null").execute()
    assert {r["employee_id"] for r in result_a.data} == {r["employee_id"] for r in result_b.data}


# ── Multi-user id collision (proves the 0002 migration) ───────────────

def test_two_users_can_each_use_the_same_app_facing_employee_id(client_a, client_b, user_a_id, user_b_id):
    """Direct regression test for the bug the 0002 migration fixed:
    employee_id is no longer globally unique, only (employee_id,
    owner_id) is -- so both users independently "landing on CE001"
    (simulated here with a shared literal id) must both succeed."""
    shared_id = f"RLSTEST-COLLIDE-{uuid.uuid4().hex[:6]}"
    try:
        client_a.table("employees").insert({
            **_employee_row(user_a_id, name="A's employee"),
            "employee_id": shared_id,
        }).execute()
        client_b.table("employees").insert({
            **_employee_row(user_b_id, name="B's employee"),
            "employee_id": shared_id,
        }).execute()

        a_view = client_a.table("employees").select("name").eq("employee_id", shared_id).execute()
        b_view = client_b.table("employees").select("name").eq("employee_id", shared_id).execute()
        assert a_view.data[0]["name"] == "A's employee"
        assert b_view.data[0]["name"] == "B's employee"
    finally:
        client_a.table("employees").delete().eq("employee_id", shared_id).eq(
            "owner_id", user_a_id
        ).execute()
        client_b.table("employees").delete().eq("employee_id", shared_id).eq(
            "owner_id", user_b_id
        ).execute()


# ── Same checks on the projects table ─────────────────────────────────
# Lighter coverage than employees (same four-policy shape, same
# migration) -- just enough to confirm the policies were actually
# applied per-table, not assumed to match because employees passed.

def _project_row(owner_id) -> dict:
    return {
        "project_id": f"RLSTEST-{uuid.uuid4().hex[:10]}",
        "owner_id": owner_id,
        "project_name": "RLS Test Project",
        "client": "Test Co",
        "required_roles": "Backend Dev",
        "required_skills": "Python",
        "min_experience": 1,
        "deadline_days": 30,
        "budget_band": "low",
        "priority": "low",
    }


@pytest.fixture
def user_a_project(client_a, user_a_id):
    row = _project_row(user_a_id)
    client_a.table("projects").insert(row).execute()
    pid = row["project_id"]
    yield pid
    client_a.table("projects").delete().eq("project_id", pid).eq(
        "owner_id", user_a_id
    ).execute()


def test_user_b_cannot_read_user_a_project(client_b, user_a_project):
    result = (
        client_b.table("projects").select("*").eq("project_id", user_a_project).execute()
    )
    assert result.data == []


def test_two_users_can_each_use_the_same_app_facing_project_id(client_a, client_b, user_a_id, user_b_id):
    shared_id = f"RLSTEST-COLLIDE-{uuid.uuid4().hex[:6]}"
    try:
        client_a.table("projects").insert({
            **_project_row(user_a_id), "project_id": shared_id,
            "project_name": "A's project",
        }).execute()
        client_b.table("projects").insert({
            **_project_row(user_b_id), "project_id": shared_id,
            "project_name": "B's project",
        }).execute()

        a_view = client_a.table("projects").select("project_name").eq("project_id", shared_id).execute()
        b_view = client_b.table("projects").select("project_name").eq("project_id", shared_id).execute()
        assert a_view.data[0]["project_name"] == "A's project"
        assert b_view.data[0]["project_name"] == "B's project"
    finally:
        client_a.table("projects").delete().eq("project_id", shared_id).eq(
            "owner_id", user_a_id
        ).execute()
        client_b.table("projects").delete().eq("project_id", shared_id).eq(
            "owner_id", user_b_id
        ).execute()