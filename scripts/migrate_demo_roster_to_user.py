"""
One-time migration: copy the 80-employee demo roster
(data/processed/employees_with_index.csv) into yashk3404@gmail.com's
own Supabase `employees` table, so it shows up in "My Employees" as
regular, owned rows -- listable, and deletable -- instead of living
only in the shared, read-only CSV.

Why this is safe to run without a service-role / admin key:
This script signs in as the target user via normal email+password
auth (supabase.auth.sign_in_with_password), the exact same call
auth.py's login form makes. Every row it inserts sets
owner_id = auth.uid() of THAT signed-in session, which is exactly
what the employees_insert RLS policy already requires
(`with check (owner_id = auth.uid())`) -- so the public anon key is
enough, no elevated credentials needed, and RLS is never bypassed.

What does NOT come across:
cost_band and proficiency (per-skill) aren't columns in the
`employees` table (see supabase/migrations/0001_schema_and_rls.sql) --
same limitation any manually-added custom employee already has via
employee_store.save_employee(). They'll show as blank/None for these
80 going forward. Not a bug introduced by this script; a pre-existing
schema gap this script doesn't try to silently fix.

Usage:
    python scripts/migrate_demo_roster_to_user.py
    python scripts/migrate_demo_roster_to_user.py --dry-run
    python scripts/migrate_demo_roster_to_user.py --email someone@else.com

Reads SUPABASE_URL / SUPABASE_KEY the same way src/auth.py does: env
vars first, falling back to .streamlit/secrets.toml. Safe to re-run --
already-migrated employee_ids for this user are skipped, not
duplicated (checked against the DB, not assumed from a local log).
"""
import argparse
import getpass
import os
import sys
from pathlib import Path

import pandas as pd
from supabase import create_client

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "processed" / "employees_with_index.csv"
TARGET_EMAIL = "yashk3404@gmail.com"
CHUNK_SIZE = 50


def _get_credentials() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        secrets_path = ROOT / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            try:
                import tomllib  # Python 3.11+
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
            except ModuleNotFoundError:
                import toml
                secrets = toml.load(secrets_path)
            url = url or secrets.get("SUPABASE_URL")
            key = key or secrets.get("SUPABASE_KEY")
    if not url or not key:
        sys.exit(
            "SUPABASE_URL / SUPABASE_KEY not found in the environment "
            "or .streamlit/secrets.toml -- set them the same way "
            "run_dashboard.bat / Streamlit Cloud's secrets manager does."
        )
    return url, key


def _row_to_employee_dict(row: pd.Series, owner_id: str) -> dict:
    skills_raw = row.get("skills", "")
    skills_list = [s.strip() for s in str(skills_raw).split(";") if s.strip()]
    exp = row.get("experience_years")
    avail = row.get("availability_pct")
    return {
        "employee_id":      row["employee_id"],
        "owner_id":         owner_id,
        "name":             row.get("name"),
        "role":             row.get("role"),
        # cast off numpy scalar types (int64/float64 aren't
        # JSON-serializable by the supabase-py client's request
        # encoder) -- plain Python int/float instead.
        "experience_years": int(exp) if pd.notna(exp) else None,
        "availability_pct": int(avail) if pd.notna(avail) else None,
        "skills":           skills_list,
        "department":       row.get("department") if pd.notna(row.get("department")) else None,
        "location":         row.get("location") if pd.notna(row.get("location")) else None,
        "github_username":  None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", default=TARGET_EMAIL,
                         help=f"Account to migrate into (default: {TARGET_EMAIL})")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what would be inserted, without writing anything.")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        sys.exit(f"Roster not found at {CSV_PATH}")

    url, key = _get_credentials()
    supabase = create_client(url, key)

    password = getpass.getpass(f"Password for {args.email}: ")
    try:
        auth_result = supabase.auth.sign_in_with_password(
            {"email": args.email, "password": password}
        )
    except Exception as e:
        sys.exit(f"Login failed for {args.email}: {e}")

    owner_id = auth_result.user.id
    print(f"Signed in as {args.email} ({owner_id})")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")

    existing = (
        supabase.table("employees")
        .select("employee_id")
        .eq("owner_id", owner_id)
        .execute()
    )
    already = {r["employee_id"] for r in existing.data}
    if already:
        print(f"{len(already)} employee(s) already present for this "
              f"user -- will be skipped, not duplicated.")

    rows = [
        _row_to_employee_dict(row, owner_id)
        for _, row in df.iterrows()
        if row["employee_id"] not in already
    ]

    print(f"{len(rows)} new row(s) to insert.")
    if not rows:
        print("Nothing to do.")
        return

    if args.dry_run:
        print("Dry run -- nothing written. Example row:")
        print(rows[0])
        return

    for i in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[i:i + CHUNK_SIZE]
        supabase.table("employees").upsert(
            chunk, on_conflict="employee_id,owner_id"
        ).execute()
        print(f"  inserted {min(i + CHUNK_SIZE, len(rows))}/{len(rows)}")

    print(
        f"\nDone. These {len(rows)} employees are now owned by "
        f"{args.email} -- they'll show up in 'My Employees' and are "
        f"eligible candidates for that account's custom projects. "
        f"Note: cost_band/per-skill proficiency did not carry over "
        f"(see this script's module docstring)."
    )


if __name__ == "__main__":
    main()
