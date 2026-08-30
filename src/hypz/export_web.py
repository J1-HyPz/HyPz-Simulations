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

from .config import DB_PATH
from .ingest import load_matches
from .models import dixon_coles

log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).resolve().parents[2] / "web" / "template.html"
PLACEHOLDER = "__MODEL_JSON__"


def build_payload(sport_id: str = "pl", season: str = "2026/27") -> dict:
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

    return {
        "evaluation": evaluation,
        "model_version": dixon_coles.MODEL_VERSION,
        "as_of": fit.as_of,
        "n_matches": fit.n_matches,
        "home_adv": round(fit.home_adv, 6),
        "rho": round(fit.rho, 6),
        "half_life_days": 270,
        "seasons": int(matches["season"].nunique()),
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
