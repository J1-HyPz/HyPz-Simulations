"""SQLite schema and helpers.

Follows section 6 of the design doc, trimmed to what Phase 1 needs. Sport-agnostic
throughout: nothing here knows what a Premier League is. JSONB becomes TEXT holding
JSON, which at this row count costs nothing.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import DB_PATH

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sports (
    sport_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS teams (
    team_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    sport_id TEXT NOT NULL REFERENCES sports(sport_id),
    name     TEXT NOT NULL,
    -- Design doc section 6: {"api_football": 33, ...}. Resolution happens once
    -- and is stored, so the fuzzy match never runs on the hot path.
    external_ids_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (sport_id, name)
);

CREATE TABLE IF NOT EXISTS games (
    game_id      TEXT PRIMARY KEY,
    sport_id     TEXT NOT NULL REFERENCES sports(sport_id),
    season       TEXT NOT NULL,
    date_utc     TEXT NOT NULL,
    home_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    status       TEXT NOT NULL DEFAULT 'scheduled',
    kickoff      TEXT,
    referee      TEXT,
    external_ids_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_games_sport_date ON games(sport_id, date_utc);
CREATE INDEX IF NOT EXISTS idx_games_season     ON games(sport_id, season);

CREATE TABLE IF NOT EXISTS game_results (
    game_id    TEXT PRIMARY KEY REFERENCES games(game_id),
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    extra_json TEXT NOT NULL DEFAULT '{}'
);

-- Section 4.4 rule 3: a stale scraper must be visible, not silent.
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    job        TEXT NOT NULL,
    sport_id   TEXT,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    status     TEXT NOT NULL,
    rows       INTEGER NOT NULL DEFAULT 0,
    error      TEXT
);

-- Section 4.4 rule 2: fetch forward from the watermark, never the whole world.
CREATE TABLE IF NOT EXISTS ingest_watermarks (
    job                     TEXT NOT NULL,
    sport_id                TEXT NOT NULL,
    last_successful_through TEXT,
    PRIMARY KEY (job, sport_id)
);

-- Starting lineups. One row per team per game; players_json holds the XI and the
-- bench. Sourced separately from results, so a missing lineup is normal rather
-- than an error.
CREATE TABLE IF NOT EXISTS lineups (
    game_id     TEXT NOT NULL REFERENCES games(game_id),
    team_id     INTEGER NOT NULL REFERENCES teams(team_id),
    formation   TEXT,
    coach       TEXT,
    players_json TEXT NOT NULL,
    source      TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (game_id, team_id)
);

-- Names the resolver could not match to a team, kept for review rather than
-- guessed at (design doc section 5).
CREATE TABLE IF NOT EXISTS unmatched_names (
    sport_id   TEXT NOT NULL,
    source     TEXT NOT NULL,
    raw_name   TEXT NOT NULL,
    context    TEXT,
    first_seen TEXT NOT NULL,
    PRIMARY KEY (sport_id, source, raw_name)
);

CREATE TABLE IF NOT EXISTS team_ratings (
    sport_id      TEXT NOT NULL,
    team_id       INTEGER NOT NULL REFERENCES teams(team_id),
    as_of         TEXT NOT NULL,
    model_version TEXT NOT NULL,
    rating_json   TEXT NOT NULL,
    PRIMARY KEY (sport_id, team_id, as_of, model_version)
);

CREATE TABLE IF NOT EXISTS forecasts (
    game_id       TEXT NOT NULL REFERENCES games(game_id),
    model_version TEXT NOT NULL,
    run_at        TEXT NOT NULL,
    inputs_hash   TEXT NOT NULL,
    home_win_prob REAL NOT NULL,
    draw_prob     REAL NOT NULL,
    away_win_prob REAL NOT NULL,
    score_dist_json TEXT NOT NULL,
    PRIMARY KEY (game_id, model_version, run_at)
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        # Additive migrations for databases created before these columns existed.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(games)")}
        for col in ("kickoff", "referee"):
            if col not in cols:
                conn.execute(f"ALTER TABLE games ADD COLUMN {col} TEXT")
        if "external_ids_json" not in cols:
            conn.execute("ALTER TABLE games ADD COLUMN external_ids_json TEXT "
                         "NOT NULL DEFAULT '{}'")
        tcols = {r[1] for r in conn.execute("PRAGMA table_info(teams)")}
        if "external_ids_json" not in tcols:
            conn.execute("ALTER TABLE teams ADD COLUMN external_ids_json TEXT "
                         "NOT NULL DEFAULT '{}'")


def upsert_sport(conn: sqlite3.Connection, sport_id: str, name: str, config: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO sports (sport_id, name, config_json) VALUES (?,?,?) "
        "ON CONFLICT(sport_id) DO UPDATE SET name=excluded.name, config_json=excluded.config_json",
        (sport_id, name, json.dumps(config)),
    )


def team_id(conn: sqlite3.Connection, sport_id: str, name: str) -> int:
    """Resolve a team name to an id, creating it on first sight."""
    conn.execute(
        "INSERT OR IGNORE INTO teams (sport_id, name) VALUES (?,?)", (sport_id, name)
    )
    row = conn.execute(
        "SELECT team_id FROM teams WHERE sport_id=? AND name=?", (sport_id, name)
    ).fetchone()
    return int(row["team_id"])


class IngestRun:
    """Context manager writing one row to ingest_runs, success or failure."""

    def __init__(self, conn: sqlite3.Connection, job: str, sport_id: str | None = None):
        self.conn, self.job, self.sport_id = conn, job, sport_id
        self.rows = 0
        self.run_id: int | None = None

    def __enter__(self) -> "IngestRun":
        cur = self.conn.execute(
            "INSERT INTO ingest_runs (job, sport_id, started_at, status) VALUES (?,?,?,?)",
            (self.job, self.sport_id, now_iso(), "running"),
        )
        self.run_id = cur.lastrowid
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = "success" if exc is None else "failed"
        self.conn.execute(
            "UPDATE ingest_runs SET ended_at=?, status=?, rows=?, error=? WHERE run_id=?",
            (now_iso(), status, self.rows, None if exc is None else str(exc)[:2000], self.run_id),
        )
        self.conn.commit()
        return False


def set_watermark(conn: sqlite3.Connection, job: str, sport_id: str, through: str) -> None:
    conn.execute(
        "INSERT INTO ingest_watermarks (job, sport_id, last_successful_through) VALUES (?,?,?) "
        "ON CONFLICT(job, sport_id) DO UPDATE SET last_successful_through=excluded.last_successful_through",
        (job, sport_id, through),
    )


def get_watermark(conn: sqlite3.Connection, job: str, sport_id: str) -> str | None:
    row = conn.execute(
        "SELECT last_successful_through FROM ingest_watermarks WHERE job=? AND sport_id=?",
        (job, sport_id),
    ).fetchone()
    return row["last_successful_through"] if row else None
