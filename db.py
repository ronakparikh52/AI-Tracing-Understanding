"""Tiny Postgres helper for the Mini-Bank sandbox.

Install:  pip install "psycopg[binary]"
"""
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

DSN = os.environ.get(
    "MINIBANK_DSN",
    "postgresql://bank:bankpassword@localhost:5433/minibank",
)


# Which database query()/execute()/connect() talk to. Defaults to `minibank`
# (steps 1-6). Step 7 points it at a per-run branch with use_database().
#
# A ContextVar, not a plain global: each THREAD gets its own value. With a
# global, 10 parallel runs would share one pointer and the last use_database()
# would win - run A's refund could land in run B's database. (Like a global,
# but every thread / asyncio task has its own copy.)
_active_dsn: ContextVar[str] = ContextVar("active_dsn", default=DSN)


@contextmanager
def connect():
    # One short-lived connection per call: simple and fine for a sandbox.
    with psycopg.connect(_active_dsn.get(), row_factory=dict_row) as conn:
        yield conn


def query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a SELECT and return rows as dicts."""
    with connect() as conn:
        return conn.execute(sql, params).fetchall()


def execute(sql: str, params: tuple | dict | None = None) -> int:
    """Run INSERT/UPDATE/DELETE, commit, and return affected row count."""
    with connect() as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount


SEED_FILE = Path(__file__).parent / "seed.sql"


def reset_db() -> None:
    """Drop all tables and re-run seed.sql - back to the pristine sandbox.

    Same result as `docker compose down -v && up`, but instant.
    """
    with connect() as conn:
        conn.execute("DROP TABLE IF EXISTS transactions, refund_policies, accounts CASCADE")
        conn.execute(SEED_FILE.read_text())
        conn.commit()


# --- Step 7: database BRANCHING with Postgres template databases -------------
# CREATE DATABASE x TEMPLATE y copies y's files on disk - a full, independent
# database in milliseconds, without replaying any SQL. So:
#   minibank_template  built once from seed.sql, never touched by an agent
#   run branches       one clone per agent run; dropped if it passed, KEPT if it
#                      failed so you can open it and see exactly what happened
TEMPLATE_DB = "minibank_template"
ADMIN_DSN = DSN.rsplit("/", 1)[0] + "/postgres"  # built-in DB, for CREATE/DROP DATABASE


def dsn_for(dbname: str) -> str:
    return DSN.rsplit("/", 1)[0] + "/" + dbname


def use_database(dbname: str) -> None:
    """Point query()/execute()/connect() at another database (e.g. a branch).

    Only affects the CURRENT thread (see _active_dsn above), so parallel runs
    each stay on their own branch.
    """
    _active_dsn.set(dsn_for(dbname))


def _admin(sql: str) -> None:
    # CREATE/DROP DATABASE can't run inside a transaction -> autocommit.
    # Database names can't be passed as %s parameters, so callers only ever
    # pass names we generate ourselves (never user/model input).
    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        conn.execute(sql)


def build_template() -> None:
    """(Re)create minibank_template from seed.sql. Always clean, by construction."""
    _admin(f"DROP DATABASE IF EXISTS {TEMPLATE_DB} WITH (FORCE)")
    _admin(f"CREATE DATABASE {TEMPLATE_DB}")
    # Seed it, then CLOSE the connection: Postgres refuses to clone a template
    # that anyone is connected to.
    with psycopg.connect(dsn_for(TEMPLATE_DB)) as conn:
        conn.execute(SEED_FILE.read_text())
        conn.commit()


# Postgres rejects a clone while another session is touching the template, and
# simultaneous clones of one template can trip over each other. A clone takes
# milliseconds, so doing them one at a time costs nothing and removes the race.
_clone_lock = threading.Lock()


def branch_db(name: str) -> None:
    """Clone the template into a fresh database called `name`."""
    _admin(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
    with _clone_lock:
        _admin(f"CREATE DATABASE {name} TEMPLATE {TEMPLATE_DB}")


def drop_branch(name: str) -> None:
    _admin(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


def list_branches(prefix: str) -> list[str]:
    with psycopg.connect(ADMIN_DSN) as conn:
        rows = conn.execute("SELECT datname FROM pg_database WHERE datname LIKE %s ORDER BY 1",
                            (prefix + "%",)).fetchall()
    return [r[0] for r in rows]


if __name__ == "__main__":
    for table in ("accounts", "transactions", "refund_policies"):
        n = query(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]
        print(f"{table:16} {n} rows")
