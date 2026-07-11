-- Item 21: Schema + Row-Level Security
--
-- Scope (Option B): only the custom layer moves here. The 80-employee
-- demo roster stays in employees_with_index.csv + precomputed .npy
-- embeddings, untouched. These tables hold ONLY user-added
-- employees/projects/assignments, keyed the same way the session-state
-- version already does (CE0xx / C0xx ids from employee_store.py /
-- project_store.py), plus one owner_id column that CSV rows will never
-- have.
--
-- Review this file line-by-line before it touches anything but a
-- throwaway test project. Nothing in items 22+ should start until
-- this is solid.

-- ============================================================
-- employees
-- ============================================================
-- Mirrors the record shape save_employee() already accepts:
-- name, role, experience_years, availability_pct, skills (list),
-- department, location. employee_id is the CE0xx id assigned by
-- _next_custom_employee_id() -- generated app-side, not by the DB,
-- so the existing id scheme doesn't have to change.

create table if not exists employees (
    employee_id       text primary key,
    owner_id          uuid references auth.users(id) on delete cascade,
    name              text not null,
    role              text not null,
    experience_years  numeric not null,
    availability_pct  numeric not null,
    skills            jsonb not null default '[]'::jsonb,
    department        text,
    location           text,
    github_username   text,  -- unused until v4, added now so v4 doesn't need a migration
    created_at        timestamptz not null default now(),

    constraint employees_experience_years_check check (experience_years >= 0),
    constraint employees_availability_pct_check check (availability_pct between 0 and 100)
);

comment on column employees.owner_id is
    'NULL = shared demo data (not actually used for this table today -- '
    'the demo roster stays in CSV -- kept nullable for schema symmetry '
    'and in case a v4+ decision moves the CSV roster in here too).';

-- ============================================================
-- projects
-- ============================================================
-- Mirrors save_project()'s expected shape: project_name, client,
-- required_roles / required_skills (";"-separated strings -- kept as
-- text, not array, so the store internals rewrite in item 23 doesn't
-- also have to change every caller's parsing convention),
-- min_experience, deadline_days, budget_band, priority.
--
-- "assignments" is deliberately NOT a column here -- it moves to its
-- own table below so RLS and CHECK constraints can apply per-row
-- instead of inside an opaque jsonb blob.

create table if not exists projects (
    project_id        text primary key,
    owner_id          uuid references auth.users(id) on delete cascade,
    project_name      text not null,
    client            text,
    required_roles    text not null,   -- ";"-separated, same convention as today
    required_skills   text,            -- ";"-separated
    min_experience    numeric,
    deadline_days     integer,
    budget_band       text,
    priority          text,
    created_at        timestamptz not null default now(),

    constraint projects_min_experience_check check (min_experience is null or min_experience >= 0),
    constraint projects_deadline_days_check check (deadline_days is null or deadline_days >= 0)
);

-- ============================================================
-- assignments
-- ============================================================
-- Replaces the {role: employee_id} dict update_project_assignments()
-- writes onto a custom project record. One row per (project, role).
-- owner_id is denormalized onto this table too (not just derived via
-- a join to projects) so its RLS policy doesn't have to subquery
-- projects on every read -- keeps the policy itself simple to audit.

create table if not exists assignments (
    id           uuid primary key default gen_random_uuid(),
    project_id   text not null references projects(project_id) on delete cascade,
    role         text not null,
    employee_id  text not null,   -- E0xx (demo roster) or CE0xx (custom) -- not FK'd to
                                   -- employees, since demo-roster ids live only in the CSV
    owner_id     uuid references auth.users(id) on delete cascade,
    created_at   timestamptz not null default now(),

    constraint assignments_project_role_unique unique (project_id, role)
);

-- ============================================================
-- Row-Level Security
-- ============================================================
-- Enforced at the DB level so the public anon key can't be tricked
-- into leaking or corrupting another user's rows regardless of what
-- app code does or fails to do.

alter table employees   enable row level security;
alter table projects    enable row level security;
alter table assignments enable row level security;

-- --- employees ---

-- Everyone authenticated can read shared demo rows (owner_id IS NULL)
-- plus their own rows. (Demo employee rows won't actually be inserted
-- into this table under Option B -- this policy exists for schema
-- symmetry / a clean v4 path, not because it's exercised today.)
create policy employees_select on employees
    for select
    using (owner_id is null or owner_id = auth.uid());

-- Users can only insert rows they own -- can't insert a NULL-owner
-- (demo) row or a row owned by someone else.
create policy employees_insert on employees
    for insert
    with check (owner_id = auth.uid());

-- Users can only update/delete their own rows -- demo rows (owner_id
-- IS NULL) are read-only to everyone, matching item 24's exit
-- criteria ("visible to all, not writable by them").
create policy employees_update on employees
    for update
    using (owner_id = auth.uid())
    with check (owner_id = auth.uid());

create policy employees_delete on employees
    for delete
    using (owner_id = auth.uid());

-- --- projects --- (same four-policy shape as employees)

create policy projects_select on projects
    for select
    using (owner_id is null or owner_id = auth.uid());

create policy projects_insert on projects
    for insert
    with check (owner_id = auth.uid());

create policy projects_update on projects
    for update
    using (owner_id = auth.uid())
    with check (owner_id = auth.uid());

create policy projects_delete on projects
    for delete
    using (owner_id = auth.uid());

-- --- assignments --- (same shape again)

create policy assignments_select on assignments
    for select
    using (owner_id is null or owner_id = auth.uid());

create policy assignments_insert on assignments
    for insert
    with check (owner_id = auth.uid());

create policy assignments_update on assignments
    for update
    using (owner_id = auth.uid())
    with check (owner_id = auth.uid());

create policy assignments_delete on assignments
    for delete
    using (owner_id = auth.uid());

-- ============================================================
-- Indexes
-- ============================================================
-- Every RLS policy above filters on owner_id -- index it on all
-- three tables so that filter doesn't do a seq scan as row counts grow.

create index if not exists employees_owner_id_idx   on employees(owner_id);
create index if not exists projects_owner_id_idx    on projects(owner_id);
create index if not exists assignments_owner_id_idx on assignments(owner_id);
create index if not exists assignments_project_id_idx on assignments(project_id);