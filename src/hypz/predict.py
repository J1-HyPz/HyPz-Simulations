"""Forecast the scheduled fixtures and record what was said, before kickoff.

This is the part that makes the pipeline worth running unattended. A backtest is
retrospective by construction; a row written to `forecasts` at 05:00 and scored
after the match is prospective evidence that cannot be contaminated by hindsight.
"""
from __future__ import annotations

import hashlib
import json
import logging

import numpy as np

from .db import IngestRun, connect, now_iso
from .ingest import load_matches
from .models import dixon_coles

log = logging.getLogger(__name__)


def _inputs_hash(fit, home: str, away: str) -> str:
    """Fingerprint every input to this forecast (design doc section 6).

    If the hash is unchanged there is nothing new to say, so the re-run is skipped.
    That is what makes a frequent schedule cheap.
    """
    h = fit.teams.index(home)
    a = fit.teams.index(away)
    payload = json.dumps([
        dixon_coles.MODEL_VERSION, fit.as_of,
        round(float(fit.attack[h]), 6), round(float(fit.defence[h]), 6),
        round(float(fit.attack[a]), 6), round(float(fit.defence[a]), 6),
        round(fit.home_adv, 6), round(fit.rho, 6),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def forecast_scheduled(sport_id: str = "pl", half_life: float | None = None) -> int:
    """Forecast every game still marked scheduled. Idempotent via inputs_hash."""
    matches = load_matches(sport_id)
    if matches.empty:
        log.warning("no matches to fit on")
        return 0
    kw = {"half_life_days": half_life} if half_life else {}
    fit = dixon_coles.fit(matches, **kw)

    with connect() as conn:
        with IngestRun(conn, job="model.forecast", sport_id=sport_id) as run:
            rows = conn.execute(
                "SELECT g.game_id, g.date_utc, th.name h, ta.name a FROM games g "
                "JOIN teams th ON th.team_id=g.home_team_id "
                "JOIN teams ta ON ta.team_id=g.away_team_id "
                "WHERE g.sport_id=? AND g.status='scheduled' ORDER BY g.date_utc",
                (sport_id,)).fetchall()
            skipped = 0
            for r in rows:
                try:
                    ih = _inputs_hash(fit, r["h"], r["a"])
                except ValueError:
                    log.warning("unknown team in %s vs %s, skipping", r["h"], r["a"])
                    continue
                prev = conn.execute(
                    "SELECT inputs_hash FROM forecasts WHERE game_id=? AND model_version=? "
                    "ORDER BY run_at DESC LIMIT 1",
                    (r["game_id"], dixon_coles.MODEL_VERSION)).fetchone()
                if prev and prev["inputs_hash"] == ih:
                    skipped += 1
                    continue
                ph, pdw, pa = fit.outcome_probs(r["h"], r["a"])
                m = fit.score_matrix(r["h"], r["a"])
                top = sorted(
                    ((int(i), int(j), float(m[i, j])) for i in range(m.shape[0])
                     for j in range(m.shape[1])), key=lambda t: -t[2])[:8]
                conn.execute(
                    "INSERT OR REPLACE INTO forecasts (game_id, model_version, run_at,"
                    " inputs_hash, home_win_prob, draw_prob, away_win_prob, score_dist_json)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (r["game_id"], dixon_coles.MODEL_VERSION, now_iso(), ih,
                     ph, pdw, pa, json.dumps([{"h": i, "a": j, "p": round(p, 6)}
                                              for i, j, p in top])))
                run.rows += 1
            log.info("forecast %d fixtures (%d unchanged, skipped)", run.rows, skipped)
            return run.rows


def scored_forecasts(sport_id: str = "pl") -> list[dict]:
    """Forecasts whose match has since been played - the live track record."""
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT f.game_id, f.run_at, f.home_win_prob, f.draw_prob, f.away_win_prob,"
            "       g.date_utc, th.name h, ta.name a, r.home_score, r.away_score "
            "FROM forecasts f JOIN games g USING(game_id) "
            "JOIN game_results r ON r.game_id=g.game_id "
            "JOIN teams th ON th.team_id=g.home_team_id "
            "JOIN teams ta ON ta.team_id=g.away_team_id "
            "WHERE g.sport_id=? ORDER BY g.date_utc DESC", (sport_id,))]
