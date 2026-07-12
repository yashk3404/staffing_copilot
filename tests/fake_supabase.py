"""
In-memory stand-in for supabase-py's Client -- item 23's test double.

Mimics just enough of the fluent query-builder surface
(table().select/insert/upsert/delete/eq/limit/execute) for
employee_store.py / project_store.py to run against headlessly, with
no network call and no live Supabase project. Enforces the same
composite-uniqueness rule the real schema does (item 23's 0002
migration: (employee_id, owner_id) / (project_id, owner_id) unique,
not employee_id/project_id alone) so a test that accidentally
reintroduces the old collision bug fails loudly here instead of only
in production.
"""
import types
import uuid


class FakeQueryError(Exception):
    pass


def _matches(row: dict, filters: dict) -> bool:
    return all(row.get(k) == v for k, v in filters.items())


class _FakeQuery:
    def __init__(self, table):
        self._table = table
        self._op = None
        self._filters = {}
        self._payload = None
        self._limit = None
        self._on_conflict = None

    def select(self, _cols="*"):
        self._op = "select"
        return self

    def insert(self, row):
        self._op = "insert"
        self._payload = row if isinstance(row, list) else [row]
        return self

    def upsert(self, row, on_conflict=None):
        self._op = "upsert"
        self._payload = row if isinstance(row, list) else [row]
        self._on_conflict = on_conflict
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def is_(self, col, val):
        self._filters[col] = val
        return self

    def limit(self, n):
        self._limit = n
        return self

    def execute(self):
        store = self._table.store

        if self._op == "select":
            rows = [dict(r) for r in store if _matches(r, self._filters)]
            if self._limit is not None:
                rows = rows[: self._limit]
            return types.SimpleNamespace(data=rows)

        if self._op in ("insert", "upsert"):
            conflict_cols = (
                self._on_conflict.split(",") if self._on_conflict else None
            )
            written = []
            for payload_row in self._payload:
                row = dict(payload_row)
                row.setdefault("id", str(uuid.uuid4()))

                existing = None
                if conflict_cols and self._op == "upsert":
                    existing = next(
                        (
                            r for r in store
                            if all(r.get(c) == row.get(c) for c in conflict_cols)
                        ),
                        None,
                    )
                elif self._table.natural_key and self._op == "upsert":
                    existing = next(
                        (
                            r for r in store
                            if all(r.get(c) == row.get(c) for c in self._table.natural_key)
                        ),
                        None,
                    )

                if existing is not None:
                    existing.update(row)
                    written.append(dict(existing))
                    continue

                # Emulate the real (col, owner_id) unique constraint --
                # a plain insert (or an upsert that didn't find a
                # match above) that collides should fail, same as
                # Postgres would.
                if self._table.natural_key:
                    collision = any(
                        all(r.get(c) == row.get(c) for c in self._table.natural_key)
                        for r in store
                    )
                    if collision:
                        raise FakeQueryError(
                            f"duplicate key value violates unique constraint "
                            f"on {self._table.name}{tuple(self._table.natural_key)}"
                        )

                store.append(row)
                written.append(dict(row))
            return types.SimpleNamespace(data=written)

        if self._op == "delete":
            to_remove = [r for r in store if _matches(r, self._filters)]
            for r in to_remove:
                store.remove(r)
            return types.SimpleNamespace(data=to_remove)

        raise FakeQueryError(f"no operation set before execute() on {self._table.name}")


class _FakeTable:
    def __init__(self, name, store, natural_key=None):
        self.name = name
        self.store = store
        self.natural_key = natural_key  # columns that must be unique together

    def select(self, cols="*"):
        return _FakeQuery(self).select(cols)

    def insert(self, row):
        return _FakeQuery(self).insert(row)

    def upsert(self, row, on_conflict=None):
        return _FakeQuery(self).upsert(row, on_conflict=on_conflict)

    def delete(self):
        return _FakeQuery(self).delete()


class FakeSupabaseClient:
    """One fresh instance per test -- tables reset with it, so no
    cross-test leakage the way _reset_session_state used to handle
    for the old session-state version."""

    _NATURAL_KEYS = {
        "employees": ("employee_id", "owner_id"),
        "projects": ("project_id", "owner_id"),
        "assignments": None,  # no app-level natural key beyond its uuid id
    }

    def __init__(self):
        self._data = {name: [] for name in self._NATURAL_KEYS}

    def table(self, name):
        return _FakeTable(name, self._data[name], self._NATURAL_KEYS[name])