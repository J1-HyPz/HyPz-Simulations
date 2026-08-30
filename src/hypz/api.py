"""Read-only HTTP API and UI (design doc section 10).

Reads only from the database. It never ingests and never refits: the ratings are
written by the scheduled `model.ratings` job and reloaded here, so a slow or
broken source can never make a page request hang. Section 3's boundary means this
process is also fine to run while the scheduler is down, and vice versa.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from . import health as health_mod
from . import ratings
from .db import connect
from .export_web import build_payload

log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).resolve().parents[2] / "web" / "template.html"
SPORT = os.environ.get("HYPZ_SPORT", "pl")

app = FastAPI(title="HyPz Simulations", docs_url="/api/docs", redoc_url=None)

# Ratings change once a night at most; reloading per request would be pure waste.
_cache: dict = {"fit": None, "as_of": None}


def _fit():
    fit = ratings.load(SPORT)
    if fit is None:
        raise HTTPException(503, "no ratings stored yet - run `hypz fit` or wait for "
                                 "the scheduled model.ratings job")
    if _cache["as_of"] != fit.as_of:
        _cache.update(fit=fit, as_of=fit.as_of)
        log.info("loaded ratings as of %s", fit.as_of)
    return _cache["fit"]


@app.get("/", response_class=HTMLResponse)
def index():
    """The forecaster page, built from stored ratings rather than a fresh fit."""
    try:
        payload = build_payload(SPORT, from_db=True)
    except Exception as exc:
        raise HTTPException(503, f"cannot build page: {exc}") from exc
    import json
    html = TEMPLATE.read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__MODEL_JSON__", json.dumps(payload)))


@app.get("/api/health")
def api_health():
    rows = health_mod.summary()
    state = health_mod.overall(rows)
    return JSONResponse({"state": state, "jobs": rows},
                        status_code=200 if state == "ok" else 503)


@app.get("/api/teams")
def api_teams(min_weight: float = 5.0):
    fit = _fit()
    return {"as_of": fit.as_of, "teams": [
        {"name": t, "attack": round(float(fit.attack[i]), 6),
         "defence": round(float(fit.defence[i]), 6),
         "weight": round(float(fit.eff_weight[i]), 2)}
        for i, t in enumerate(fit.teams) if fit.eff_weight[i] >= min_weight]}


@app.get("/api/forecast")
def api_forecast(home: str = Query(...), away: str = Query(...)):
    """Forecast any matchup. Closed form, so this is microseconds, not a simulation."""
    fit = _fit()
    if home == away:
        raise HTTPException(400, "home and away must differ")
    for t in (home, away):
        if t not in fit.teams:
            raise HTTPException(404, f"unknown team: {t}")
    h, d, a = fit.outcome_probs(home, away)
    lam, mu = fit.rates(home, away)
    m = fit.score_matrix(home, away)
    top = sorted(((int(i), int(j), float(m[i, j]))
                  for i in range(m.shape[0]) for j in range(m.shape[1])),
                 key=lambda t: -t[2])[:10]
    return {
        "home": home, "away": away, "as_of": fit.as_of,
        "probabilities": {"home": h, "draw": d, "away": a},
        "expected_goals": {"home": lam, "away": mu},
        "likeliest": [{"home": i, "away": j, "p": p} for i, j, p in top],
    }


@app.get("/api/fixtures")
def api_fixtures():
    with connect() as conn:
        rows = conn.execute(
            "SELECT g.date_utc, th.name home, ta.name away, g.status,"
            "       f.home_win_prob, f.draw_prob, f.away_win_prob, f.run_at "
            "FROM games g "
            "JOIN teams th ON th.team_id=g.home_team_id "
            "JOIN teams ta ON ta.team_id=g.away_team_id "
            "LEFT JOIN forecasts f ON f.game_id=g.game_id "
            "WHERE g.sport_id=? AND g.status='scheduled' ORDER BY g.date_utc",
            (SPORT,)).fetchall()
    return {"fixtures": [dict(r) for r in rows]}


@app.get("/api/evaluation")
def api_evaluation():
    import json
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM model_evaluations WHERE sport_id=? ORDER BY eval_id DESC LIMIT 1",
            (SPORT,)).fetchone()
    if row is None:
        raise HTTPException(404, "no evaluation recorded - run `hypz backtest --save`")
    return {"window": row["window"], "n": row["n"], "brier": row["brier"],
            "log_loss": row["log_loss"], "accuracy": row["accuracy"],
            "calibration": json.loads(row["calibration_json"]),
            "baselines": json.loads(row["baselines_json"])}
