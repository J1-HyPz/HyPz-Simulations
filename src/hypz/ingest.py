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
from .db import (IngestRun, connect, get_watermark, now_iso, set_watermark,
                 team_id, upsert_sport)

log = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(s: str) -> str:
    return _SLUG.sub("-", s.lower()).strip("-")


def game_key(sport_id: str, rec: GameRecord) -> str:
    """Deterministic natural key - the same fixture always maps to the same id,
    which is what makes re-running an ingest safe."""
    return f"{sport_id}:{rec.season}:{rec.date_utc}:{_slug(rec.home_team)}:{_slug(rec.away_team)}"


def _upsert_game(conn, cfg, rec: GameRecord) -> str:
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
    return gid


def ingest_fixtures(adapter: SportAdapter) -> int:
    """Upcoming fixtures. Safe to run repeatedly; a fixture that has since been
    played is simply overwritten by the results job on the same key."""
    cfg = adapter.config
    with connect() as conn:
        upsert_sport(conn, cfg.sport_id, cfg.name,
                     {"simulation_unit": cfg.simulation_unit, **cfg.extra})
        with IngestRun(conn, job="ingest.fixtures", sport_id=cfg.sport_id) as run:
            for rec in adapter.fetch_fixtures():
                # Never downgrade a finished match back to scheduled.
                gid = game_key(cfg.sport_id, rec)
                row = conn.execute("SELECT status FROM games WHERE game_id=?", (gid,)).fetchone()
                if row and row["status"] == "final":
                    continue
                _upsert_game(conn, cfg, rec)
                run.rows += 1
            log.info("ingested %d fixtures for %s", run.rows, cfg.sport_id)
            return run.rows


def _complete_through(conn, cfg, current_year: int) -> int | None:
    """Highest season-start year such that every season up to it is finished.

    "Finished" means the season has ended, not that its match count is full.
    Counting was the obvious rule and it is wrong: football-data.co.uk ships only
    335 of 380 matches for 2003/04 and 2004/05, so a count test can never mark
    them complete and the watermark would refetch two decades forever.

    Contiguity still applies - a season missing outright blocks the watermark,
    because that is a hole we could still fill.
    """
    seasons = {int(r["season"].split("/")[0]) for r in conn.execute(
        "SELECT DISTINCT season FROM games WHERE sport_id=? AND status='final'",
        (cfg.sport_id,))}
    if not seasons:
        return None
    year, best = min(seasons), None
    while year in seasons and year < current_year:
        best = year
        year += 1
    return best


def ingest_results(adapter: SportAdapter, seasons: list[str] | None = None,
                   force: bool = False) -> int:
    cfg = adapter.config
    with connect() as conn:
        upsert_sport(conn, cfg.sport_id, cfg.name, {"simulation_unit": cfg.simulation_unit, **cfg.extra})
        with IngestRun(conn, job="ingest.results", sport_id=cfg.sport_id) as run:
            if seasons is None and not force:
                # Design doc section 4.4 rule 2: fetch forward from the watermark.
                wm = get_watermark(conn, "ingest.results", cfg.sport_id)
                try:
                    year = int(wm) if wm else None
                except ValueError:
                    # An older build stored a date here. Treat anything unparseable
                    # as absent and fall through to a full fetch, which rewrites it.
                    log.warning("ignoring unrecognised watermark %r", wm)
                    year = None
                if year is not None:
                    seasons = adapter.seasons_after(year)
                    log.info("watermark %s: fetching %d season(s) forward", year, len(seasons))
            records = adapter.fetch_results(seasons)
            for rec in records:
                gid = _upsert_game(conn, cfg, rec)
                if rec.is_played:
                    conn.execute(
                        "INSERT INTO game_results (game_id, home_score, away_score, extra_json) "
                        "VALUES (?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET "
                        "home_score=excluded.home_score, away_score=excluded.away_score, "
                        "extra_json=excluded.extra_json",
                        (gid, rec.home_score, rec.away_score, json.dumps(rec.extra)),
                    )
                run.rows += 1
            done = _complete_through(conn, cfg, adapter.current_season_year())
            if done is not None:
                set_watermark(conn, "ingest.results", cfg.sport_id, str(done))
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
