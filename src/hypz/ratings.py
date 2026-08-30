"""Persist and reload a fitted model.

The serving layer must never refit on request (design doc section 10), so the fit
is written to `team_ratings` by a scheduled job and read back by anything that
serves. That is also what keeps section 3's boundary honest: the API touches the
database and nothing else.
"""
from __future__ import annotations

import json
import logging

import numpy as np

from .db import connect
from .models.dixon_coles import MODEL_VERSION, DixonColesFit

log = logging.getLogger(__name__)

GLOBAL_KEY = MODEL_VERSION + ":global"


def persist(fit: DixonColesFit, sport_id: str = "pl") -> int:
    with connect() as conn:
        ids = {r["name"]: r["team_id"] for r in conn.execute(
            "SELECT team_id, name FROM teams WHERE sport_id=?", (sport_id,))}
        n = 0
        for i, team in enumerate(fit.teams):
            tid = ids.get(team)
            if tid is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO team_ratings "
                "(sport_id, team_id, as_of, model_version, rating_json) VALUES (?,?,?,?,?)",
                (sport_id, tid, fit.as_of, MODEL_VERSION, json.dumps({
                    "name": team,
                    "attack": float(fit.attack[i]),
                    "defence": float(fit.defence[i]),
                    "weight": float(fit.eff_weight[i]),
                })))
            n += 1
        # team_id 0 is not a real team; it carries the fit-wide parameters.
        conn.execute(
            "INSERT OR REPLACE INTO team_ratings "
            "(sport_id, team_id, as_of, model_version, rating_json) VALUES (?,?,?,?,?)",
            (sport_id, 0, fit.as_of, GLOBAL_KEY, json.dumps({
                "home_adv": fit.home_adv, "rho": fit.rho,
                "n_matches": fit.n_matches, "log_likelihood": fit.log_likelihood,
            })))
        log.info("persisted ratings for %d teams as of %s", n, fit.as_of)
        return n


def load(sport_id: str = "pl") -> DixonColesFit | None:
    """Rebuild the most recent fit from the database. None if none stored."""
    with connect() as conn:
        g = conn.execute(
            "SELECT as_of, rating_json FROM team_ratings WHERE sport_id=? AND model_version=? "
            "ORDER BY as_of DESC LIMIT 1", (sport_id, GLOBAL_KEY)).fetchone()
        if g is None:
            return None
        as_of = g["as_of"]
        meta = json.loads(g["rating_json"])
        rows = conn.execute(
            "SELECT rating_json FROM team_ratings WHERE sport_id=? AND model_version=? "
            "AND as_of=? AND team_id != 0", (sport_id, MODEL_VERSION, as_of)).fetchall()

    entries = sorted((json.loads(r["rating_json"]) for r in rows), key=lambda d: d["name"])
    if not entries:
        return None
    return DixonColesFit(
        teams=[e["name"] for e in entries],
        attack=np.array([e["attack"] for e in entries]),
        defence=np.array([e["defence"] for e in entries]),
        home_adv=float(meta["home_adv"]),
        rho=float(meta["rho"]),
        as_of=as_of,
        n_matches=int(meta.get("n_matches", 0)),
        log_likelihood=float(meta.get("log_likelihood", 0.0)),
        eff_weight=np.array([e.get("weight", 0.0) for e in entries]),
    )
