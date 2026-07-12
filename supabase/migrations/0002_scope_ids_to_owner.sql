-- Item 23: fix employee_id / project_id uniqueness for multi-user use
--
-- Item 21's schema made employee_id / project_id the sole PRIMARY KEY
-- on their tables. That's fine with a single test user, but
-- employee_store.py / project_store.py generate those ids
-- sequentially ("CE001", "CE002", ...) per owner -- and RLS means a
-- user can only ever SELECT their own rows (owner_id = auth.uid()) to
-- compute "next id." Two different users each saving their first
-- custom employee would both independently land on "CE001," and the
-- second INSERT would fail outright on the primary key collision.
--
-- Fix: employee_id/project_id become per-owner identifiers, not
-- globally unique ones. A surrogate uuid `id` becomes the real
-- primary key; (employee_id, owner_id) / (project_id, owner_id)
-- become composite UNIQUE constraints instead. Two different users
-- can now legitimately each have a "CE001" or a "C001".
--
-- Written idempotently (IF EXISTS / IF NOT EXISTS throughout) so a
-- retry after a partial failure is safe to just re-run as-is.

-- ============================================================
-- Step 1 -- drop the dependent FK on assignments FIRST.
-- ============================================================
-- assignments_project_id_fkey depends on projects_pkey -- trying to
-- drop projects_pkey while this still exists fails with 2BP01. Has to
-- go first, gets rebuilt (composite, against the new unique
-- constraint) in step 3.
alter table assignments drop constraint if exists assignments_project_id_fkey;

-- ============================================================
-- Step 2 -- replace the natural-key primary keys with surrogate uuids.
-- ============================================================

alter table employees drop constraint if exists employees_pkey;
alter table employees add column if not exists id uuid not null default gen_random_uuid();
alter table employees add constraint employees_pkey primary key (id);
alter table employees add constraint employees_employee_owner_unique
    unique (employee_id, owner_id);

alter table projects drop constraint if exists projects_pkey;
alter table projects add column if not exists id uuid not null default gen_random_uuid();
alter table projects add constraint projects_pkey primary key (id);
alter table projects add constraint projects_project_owner_unique
    unique (project_id, owner_id);

-- ============================================================
-- Step 3 -- rebuild assignments' constraints against the new shape.
-- ============================================================

alter table assignments add constraint assignments_project_owner_fkey
    foreign key (project_id, owner_id) references projects (project_id, owner_id);

-- Two different owners can legitimately both have a "C001" with a
-- "Backend Dev" role assigned -- scope the uniqueness to owner too.
alter table assignments drop constraint if exists assignments_project_role_unique;
alter table assignments add constraint assignments_project_role_owner_unique
    unique (project_id, role, owner_id);