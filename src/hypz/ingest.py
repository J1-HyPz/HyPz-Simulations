"""Ingestion pipeline.

Obeys the three rules in design doc section 4.4: every write is an idempotent
upsert on a natural key, progress is watermarked, and every run is instrumented
so a stale source is visible rather than silent.
"""
from __future__ import annotations

import json
import logging
import re

from .adapters.base import GameRecord, SportAdapter
from .db import IngestRun, connect, now_iso, set_watermark, team_id, upsert_sport

log = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(s: str) -> str:
    return _SLUG.sub("-", s.lower()).strip("-")


def game_key(sport_id: str, rec: GameRecord) -> str:
    """Deterministic natural key - the same fixture always maps to the same id,
    which is what makes re-running an ingest safe."""
    return f"{sport_id}:{rec.season}:{rec.date_utc}:{_slug(rec.home_team)}:{_slug(rec.away_team)}"


def ingest_results(adapter: SportAdapter, seasons: list[str] | None = None) -> int:
    cfg = adapter.config
    with connect() as conn:
        upsert_sport(conn, cfg.sport_id, cfg.name, {"simulation_unit": cfg.simulation_unit, **cfg.extra})
        with IngestRun(conn, job="ingest.results", sport_id=cfg.sport_id) as run:
            records = adapter.fetch_results(seasons)
            for rec in records:
                gid = game_key(cfg.sport_id, rec)
                h = team_id(conn, cfg.sport_id, rec.home_team)
                a = team_id(conn, cfg.sport_id, rec.away_team)
                conn.execute(
                    "INSERT INTO games (game_id, sport_id, season, date_utc, home_team_id, away_team_id, status) "
                    "VALUES (?,?,?,?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET "
                    "season=excluded.season, date_utc=excluded.date_utc, status=excluded.status",
                    (gid, cfg.sport_id, rec.season, rec.date_utc, h, a,
                     "final" if rec.is_played else "scheduled"),
                )
                if rec.is_played:
                    conn.execute(
                        "INSERT INTO game_results (game_id, home_score, away_score, extra_json) "
                        "VALUES (?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET "
                        "home_score=excluded.home_score, away_score=excluded.away_score, "
                        "extra_json=excluded.extra_json",
                        (gid, rec.home_score, rec.away_score, json.dumps(rec.extra)),
                    )
                run.rows += 1
            if records:
                set_watermark(conn, "ingest.results", cfg.sport_id,
                              max(r.date_utc for r in records))
            log.info("ingested %d records for %s", run.rows, cfg.sport_id)
            return run.rows


def load_matches(sport_id: str):
    """Played matches as a DataFrame, ready for the model."""
    import pandas as pd

    with connect() as conn:
        rows = conn.execute(
            "SELECT g.game_id, g.date_utc AS date, g.season, "
            "       th.name AS home, ta.name AS away, "
            "       r.home_score, r.away_score, r.extra_json "
            "FROM games g "
            "JOIN game_results r ON r.game_id = g.game_id "
            "JOIN teams th ON th.team_id = g.home_team_id "
            "JOIN teams ta ON ta.team_id = g.away_team_id "
            "WHERE g.sport_id = ? ORDER BY g.date_utc",
            (sport_id,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])
