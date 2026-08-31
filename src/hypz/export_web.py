"""Build the standalone forecaster page.

Fits the model, serialises it, and injects it into web/template.html. The page
recomputes the Dixon-Coles scoreline grid in JavaScript rather than shipping
precomputed fixtures - the model is closed form, so every matchup is live and
the whole thing stays a single file with no backend.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from . import leagues as leagues_mod
from . import matchcard, ratings
from .config import DB_PATH
from .ingest import load_matches
from .models import dixon_coles

log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).resolve().parents[2] / "web" / "template.html"
PLACEHOLDER = "__MODEL_JSON__"


def build_payload(sport_id: str = "pl", season: str = "2026/27",
                  from_db: bool = False) -> dict:
    """Assemble everything the page needs.

    `from_db` reloads the last stored fit instead of refitting. The serving layer
    uses it so that a page request never triggers a fit (design doc section 10).
    """
    if from_db:
        fit = ratings.load(sport_id)
        if fit is None:
            raise RuntimeError("no stored ratings; run the model.ratings job first")
    else:
        matches = load_matches(sport_id)
        if matches.empty:
            raise SystemExit("no matches - run ingest first")
        fit = dixon_coles.fit(matches)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        current = sorted(
            r[0] for r in con.execute(
                "SELECT DISTINCT t.name FROM games g JOIN teams t "
                "ON t.team_id IN (g.home_team_id, g.away_team_id) WHERE g.season=?",
                (season,),
            )
        )
        played = [dict(r) for r in con.execute(
            "SELECT g.date_utc d, th.name h, ta.name a, r.home_score hs, r.away_score asc_ "
            "FROM games g JOIN game_results r USING(game_id) "
            "JOIN teams th ON th.team_id=g.home_team_id "
            "JOIN teams ta ON ta.team_id=g.away_team_id "
            "WHERE g.season=? ORDER BY g.date_utc", (season,))]
    finally:
        con.close()

    con = sqlite3.connect(DB_PATH)
    try:
        n_seasons = con.execute(
            "SELECT COUNT(DISTINCT season) FROM games WHERE sport_id=?", (sport_id,)
        ).fetchone()[0]
    finally:
        con.close()

    # Latest walk-forward evaluation, if one has been run.
    evaluation = None
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            "SELECT * FROM model_evaluations WHERE sport_id=? ORDER BY eval_id DESC LIMIT 1",
            (sport_id,)).fetchone()
        if row:
            evaluation = {
                "window": row["window"], "n": row["n"],
                "brier": row["brier"], "log_loss": row["log_loss"], "accuracy": row["accuracy"],
                "calibration": json.loads(row["calibration_json"]),
                "baselines": json.loads(row["baselines_json"]),
                "seasons": json.loads(row["seasons_json"]) if row["seasons_json"] else [],
            }
    except sqlite3.OperationalError:
        pass  # no evaluation table yet
    finally:
        con.close()

    # Upcoming fixtures with the forecast that was recorded before kickoff.
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        upcoming = [dict(r) for r in con.execute(
            "SELECT g.date_utc d, th.name h, ta.name a, f.home_win_prob hp,"
            "       f.draw_prob dp, f.away_win_prob ap, f.run_at "
            "FROM games g JOIN forecasts f USING(game_id) "
            "JOIN teams th ON th.team_id=g.home_team_id "
            "JOIN teams ta ON ta.team_id=g.away_team_id "
            "WHERE g.sport_id=? AND g.status='scheduled' "
            "ORDER BY g.date_utc, th.name", (sport_id,))]
    except sqlite3.OperationalError:
        upcoming = []
    finally:
        con.close()

    from .health import overall, summary
    try:
        jobs = summary()
        health = {"state": overall(jobs), "jobs": jobs}
    except sqlite3.OperationalError:
        health = None

    try:
        sched = matchcard.schedule_all(days=7, back=7)
    except Exception as exc:            # a page must still render without it
        log.warning("schedule unavailable: %s", exc)
        sched = None

    # Every league's ratings, so the forecaster can switch without a round trip.
    all_leagues = {}
    for lid in leagues_mod.all_ids():
        lf = ratings.load(lid)
        if lf is None:
            continue
        lg = leagues_mod.get(lid)
        con2 = sqlite3.connect(DB_PATH)
        con2.row_factory = sqlite3.Row
        try:
            cur = sorted(r[0] for r in con2.execute(
                "SELECT DISTINCT t.name FROM games g JOIN teams t "
                "ON t.team_id IN (g.home_team_id, g.away_team_id) "
                "WHERE g.sport_id=? AND g.season=?", (lid, season)))
            nseasons = con2.execute(
                "SELECT COUNT(DISTINCT season) FROM games WHERE sport_id=?", (lid,)
            ).fetchone()[0]
            ngames = con2.execute(
                "SELECT COUNT(*) FROM games WHERE sport_id=?", (lid,)).fetchone()[0]
        finally:
            con2.close()
        all_leagues[lid] = {
            "name": lg.name, "country": lg.country, "as_of": lf.as_of,
            "home_adv": round(lf.home_adv, 6), "rho": round(lf.rho, 6),
            "n_matches": lf.n_matches, "seasons": nseasons, "games": ngames,
            "current_teams": cur,
            "teams": [{"name": t,
                       "attack": round(float(lf.attack[i]), 6),
                       "defence": round(float(lf.defence[i]), 6),
                       "weight": round(float(lf.eff_weight[i]), 2)}
                      for i, t in enumerate(lf.teams)],
        }

    return {
        "leagues": all_leagues,
        "default_league": sport_id,
        "schedule": sched,
        "upcoming": upcoming,
        "health": health,
        "evaluation": evaluation,
        "model_version": dixon_coles.MODEL_VERSION,
        "as_of": fit.as_of,
        "n_matches": fit.n_matches,
        "home_adv": round(fit.home_adv, 6),
        "rho": round(fit.rho, 6),
        "half_life_days": 270,
        "seasons": n_seasons,
        "current_season": season,
        "current_teams": current,
        "played": played,
        "teams": [
            {"name": t,
             "attack": round(float(fit.attack[i]), 6),
             "defence": round(float(fit.defence[i]), 6),
             "weight": round(float(fit.eff_weight[i]), 2)}
            for i, t in enumerate(fit.teams)
        ],
    }


def export(out_path: Path, sport_id: str = "pl", season: str = "2026/27") -> Path:
    payload = build_payload(sport_id, season)
    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise SystemExit(f"{TEMPLATE} has no {PLACEHOLDER} placeholder")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html.replace(PLACEHOLDER, json.dumps(payload)), encoding="utf-8")
    log.info("wrote %s (%d bytes, %d teams)", out_path, out_path.stat().st_size, len(payload["teams"]))
    return out_path
