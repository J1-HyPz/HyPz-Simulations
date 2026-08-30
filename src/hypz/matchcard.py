"""Per-fixture context: the schedule, and what is worth knowing about each game.

Everything here comes from data we already hold. Nothing calls out to a network,
so this is safe to serve on request (design doc section 10).

Two things deliberately absent, because the source does not carry them: starting
lineups, and in-play state. See `LIMITATIONS`.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta

from . import lineups as lineups_mod
from .db import connect

log = logging.getLogger(__name__)

# Beyond this, a "last match" is a promoted side's absence from the division or a
# summer break, not rest. Coventry's previous top-flight match was 25 years ago;
# reporting 9,225 days of rest is noise, and their 2001 results are not form.
FORM_MAX_AGE_DAYS = 400
REST_MAX_DAYS = 45

LIVE_LIMITATION = {
    "live": "the results feed publishes after full time, so there is no in-play state",
}
NO_LINEUP_SOURCE = {
    "lineups": "no lineup provider configured; set HIGHLIGHTLY_KEY to enable",
}


def limitations() -> dict:
    """What this deployment cannot show, given how it is configured.

    A key being present is not the same as lineups being available: the provider's
    free plan covers only past seasons, and reporting nothing in that case would
    leave an unexplained blank.
    """
    out = dict(LIVE_LIMITATION)
    if not lineups_mod.get_provider().configured:
        out.update(NO_LINEUP_SOURCE)
    else:
        restriction = lineups_mod.plan_restriction()
        if restriction:
            out["lineups"] = f"provider plan does not cover this season ({restriction})"
    return out

_GAME_SELECT = """
SELECT g.game_id, g.date_utc, g.season, g.status, g.kickoff, g.referee,
       th.name AS home, ta.name AS away,
       r.home_score, r.away_score, r.extra_json
FROM games g
JOIN teams th ON th.team_id = g.home_team_id
JOIN teams ta ON ta.team_id = g.away_team_id
LEFT JOIN game_results r ON r.game_id = g.game_id
"""


def _outcome(gf: int, ga: int) -> str:
    return "W" if gf > ga else ("D" if gf == ga else "L")


def _recent(conn, sport_id: str, team: str, before: str, limit: int = 5,
            max_age_days: int = FORM_MAX_AGE_DAYS) -> list[dict]:
    """That team's last `limit` completed matches before `before`, newest first.

    Bounded by age: a promoted side has no recent matches in this division, and
    saying so is more use than quoting results from a previous spell.
    """
    floor = (date.fromisoformat(before) - timedelta(days=max_age_days)).isoformat()
    rows = conn.execute(_GAME_SELECT + """
        WHERE g.sport_id = ? AND g.status = 'final' AND g.date_utc < ?
          AND g.date_utc >= ? AND (th.name = ? OR ta.name = ?)
        ORDER BY g.date_utc DESC LIMIT ?""",
        (sport_id, before, floor, team, team, limit)).fetchall()
    out = []
    for r in rows:
        at_home = r["home"] == team
        gf = r["home_score"] if at_home else r["away_score"]
        ga = r["away_score"] if at_home else r["home_score"]
        out.append({
            "date": r["date_utc"], "opponent": r["away"] if at_home else r["home"],
            "venue": "H" if at_home else "A",
            "gf": gf, "ga": ga, "result": _outcome(gf, ga),
        })
    return out


def _form_summary(recent: list[dict]) -> dict:
    if not recent:
        # Newly promoted, or otherwise absent from this division recently.
        return {"played": 0, "form": "", "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0,
                "ppg": None, "note": "no recent matches in this division"}
    w = sum(1 for m in recent if m["result"] == "W")
    d = sum(1 for m in recent if m["result"] == "D")
    l = sum(1 for m in recent if m["result"] == "L")
    return {
        "played": len(recent),
        # Oldest first reads the way people say it: "WWDLW".
        "form": "".join(m["result"] for m in reversed(recent)),
        "w": w, "d": d, "l": l,
        "gf": sum(m["gf"] for m in recent),
        "ga": sum(m["ga"] for m in recent),
        "ppg": round((3 * w + d) / len(recent), 2),
    }


def _venue_record(conn, sport_id: str, team: str, season: str, venue: str, before: str) -> dict:
    """This season's record at one venue - home form and away form differ enough
    that a combined table hides the thing you actually want to know."""
    side = "th" if venue == "H" else "ta"
    rows = conn.execute(_GAME_SELECT + f"""
        WHERE g.sport_id = ? AND g.status = 'final' AND g.season = ?
          AND g.date_utc < ? AND {side}.name = ?""",
        (sport_id, season, before, team)).fetchall()
    w = d = l = gf = ga = 0
    for r in rows:
        h = venue == "H"
        f_, a_ = (r["home_score"], r["away_score"]) if h else (r["away_score"], r["home_score"])
        gf += f_; ga += a_
        res = _outcome(f_, a_)
        w += res == "W"; d += res == "D"; l += res == "L"
    return {"venue": venue, "played": len(rows), "w": w, "d": d, "l": l, "gf": gf, "ga": ga}


def _head_to_head(conn, sport_id: str, home: str, away: str, before: str, limit: int = 6) -> dict:
    rows = conn.execute(_GAME_SELECT + """
        WHERE g.sport_id = ? AND g.status = 'final' AND g.date_utc < ?
          AND ((th.name = ? AND ta.name = ?) OR (th.name = ? AND ta.name = ?))
        ORDER BY g.date_utc DESC LIMIT ?""",
        (sport_id, before, home, away, away, home, limit)).fetchall()
    meetings, hw, dr, aw = [], 0, 0, 0
    for r in rows:
        meetings.append({"date": r["date_utc"], "home": r["home"], "away": r["away"],
                         "home_score": r["home_score"], "away_score": r["away_score"]})
        # Counted from the perspective of the upcoming fixture's home side.
        if r["home"] == home:
            hw += r["home_score"] > r["away_score"]
            aw += r["away_score"] > r["home_score"]
        else:
            hw += r["away_score"] > r["home_score"]
            aw += r["home_score"] > r["away_score"]
        dr += r["home_score"] == r["away_score"]
    return {"meetings": meetings, "home_wins": hw, "draws": dr, "away_wins": aw}


def _rest_days(conn, sport_id: str, team: str, before: str) -> int | None:
    """Days since the team's previous match, or None when that gap is a season
    break or an absence from the division rather than rest."""
    row = conn.execute(_GAME_SELECT + """
        WHERE g.sport_id = ? AND g.status = 'final' AND g.date_utc < ?
          AND (th.name = ? OR ta.name = ?)
        ORDER BY g.date_utc DESC LIMIT 1""",
        (sport_id, before, team, team)).fetchone()
    if row is None:
        return None
    gap = (date.fromisoformat(before) - date.fromisoformat(row["date_utc"])).days
    return gap if gap <= REST_MAX_DAYS else None


def _forecast(conn, game_id: str) -> dict | None:
    row = conn.execute(
        "SELECT home_win_prob, draw_prob, away_win_prob, run_at, model_version "
        "FROM forecasts WHERE game_id=? ORDER BY run_at DESC LIMIT 1", (game_id,)).fetchone()
    if row is None:
        return None
    return {"home": row["home_win_prob"], "draw": row["draw_prob"],
            "away": row["away_win_prob"], "run_at": row["run_at"],
            "model_version": row["model_version"]}


def _post_match(row) -> dict | None:
    """Statistics and a factual summary, once the match has been played."""
    if row["status"] != "final" or row["home_score"] is None:
        return None
    extra = json.loads(row["extra_json"] or "{}")
    stats, ht = extra.get("stats", {}), extra.get("half_time")
    h, a, hs, as_ = row["home"], row["away"], row["home_score"], row["away_score"]

    bits = []
    if hs > as_:
        bits.append(f"{h} beat {a} {hs}–{as_}.")
    elif hs < as_:
        bits.append(f"{a} won {as_}–{hs} at {h}.")
    else:
        bits.append(f"{h} and {a} drew {hs}–{as_}.")
    if ht and "h" in ht and "a" in ht:
        lead = h if ht["h"] > ht["a"] else (a if ht["a"] > ht["h"] else None)
        bits.append(f"{lead} led {max(ht['h'], ht['a'])}–{min(ht['h'], ht['a'])} at the break."
                    if lead else f"It was {ht['h']}–{ht['a']} at half time.")
        # A side that trailed at the break and won had to come back for it.
        if lead and ((lead == h and hs < as_) or (lead == a and as_ < hs)):
            bits.append(f"{a if lead == h else h} recovered to win.")
    if "shots" in stats:
        sh = stats["shots"]
        bits.append(f"Shots finished {sh['h']}–{sh['a']}"
                    + (f", on target {stats['shots_on_target']['h']}–"
                       f"{stats['shots_on_target']['a']}." if "shots_on_target" in stats else "."))
        # Worth saying only when the shot count and the scoreline disagree.
        if "shots_on_target" in stats:
            sot = stats["shots_on_target"]
            if sh["h"] > sh["a"] * 1.5 and hs < as_:
                bits.append(f"{h} were the more dominant side and lost anyway.")
            elif sh["a"] > sh["h"] * 1.5 and as_ < hs:
                bits.append(f"{a} were the more dominant side and lost anyway.")
    reds = stats.get("reds")
    if reds and (reds["h"] or reds["a"]):
        who = h if reds["h"] else a
        bits.append(f"{who} finished with {reds['h'] or reds['a']} player(s) sent off.")
    return {"score": {"home": hs, "away": as_}, "half_time": ht,
            "stats": stats, "summary": " ".join(bits)}


def card(game_id: str, sport_id: str = "pl") -> dict | None:
    """Everything known about one fixture."""
    with connect() as conn:
        row = conn.execute(_GAME_SELECT + " WHERE g.game_id = ?", (game_id,)).fetchone()
        if row is None:
            return None
        d, home, away = row["date_utc"], row["home"], row["away"]
        extra = json.loads(row["extra_json"] or "{}") if row["extra_json"] else {}

        recent_h = _recent(conn, sport_id, home, d)
        recent_a = _recent(conn, sport_id, away, d)
        out = {
            "game_id": game_id, "date": d, "kickoff": row["kickoff"],
            "referee": row["referee"], "season": row["season"], "status": row["status"],
            "home": home, "away": away,
            "forecast": _forecast(conn, game_id),
            "pre_match": {
                "form": {
                    "home": {**_form_summary(recent_h), "matches": recent_h},
                    "away": {**_form_summary(recent_a), "matches": recent_a},
                },
                "venue_record": {
                    "home": _venue_record(conn, sport_id, home, row["season"], "H", d),
                    "away": _venue_record(conn, sport_id, away, row["season"], "A", d),
                },
                "head_to_head": _head_to_head(conn, sport_id, home, away, d),
                "rest_days": {"home": _rest_days(conn, sport_id, home, d),
                              "away": _rest_days(conn, sport_id, away, d)},
                "market": {k: extra[k] for k in ("open_h", "open_d", "open_a") if k in extra} or None,
            },
            "result": _post_match(row),
            "lineups": lineups_mod.for_game(game_id),
            "unavailable": limitations(),
        }
        return out


def schedule(sport_id: str = "pl", days: int = 7, today: str | None = None,
             back: int = 7) -> dict:
    """Fixtures around today: `back` days of results, `days` ahead of fixtures.

    Showing both is how anyone actually reads a football schedule, and it means
    recent results do not need a separate panel repeating the same rows.
    """
    now = today or date.today().isoformat()
    start = (date.fromisoformat(now) - timedelta(days=back)).isoformat()
    end = (date.fromisoformat(now) + timedelta(days=days)).isoformat()
    with connect() as conn:
        ids = [r["game_id"] for r in conn.execute(
            "SELECT game_id FROM games WHERE sport_id=? AND date_utc >= ? AND date_utc <= ? "
            "ORDER BY date_utc, kickoff", (sport_id, start, end))]
    games = [c for c in (card(g, sport_id) for g in ids) if c]
    by_day: dict[str, list] = {}
    for g in games:
        by_day.setdefault(g["date"], []).append(g)
    return {
        "from": start, "to": end, "today": now, "count": len(games),
        "days": [{"date": d, "games": by_day[d]} for d in sorted(by_day)],
        "unavailable": limitations(),
    }
