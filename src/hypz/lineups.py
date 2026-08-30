"""Starting lineups from API-Football.

Two problems to solve beyond fetching. First, the two sources name teams
differently ("Manchester United" against "Man United"), so identifiers are
resolved once by alias and fuzzy match, stored, and never recomputed; anything
unresolved goes to a queue for review rather than being guessed at (design doc
section 5). Second, the free tier allows 100 requests a day, so every call is
budgeted and a lineup is never fetched twice.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from datetime import datetime, timedelta, timezone

from .adapters import api_football as af
from .adapters import highlightly as hl
from .db import IngestRun, connect, get_state, now_iso, set_state

log = logging.getLogger(__name__)

PROVIDER_ENV = "HYPZ_LINEUP_PROVIDER"
# Records that the configured plan cannot serve the current season, so the page
# can say lineups are unavailable and why, instead of silently showing none.
PLAN_STATE_KEY = "lineups.plan_restriction"


def get_provider(name: str | None = None):
    """Pick a lineup provider.

    Default is "auto": prefer Highlightly, whose free tier covers the current
    season, and fall back to API-Football, whose free tier does not. An
    unconfigured provider is still returned so callers can print a clean message
    rather than branch on None.
    """
    import os
    name = (name or os.environ.get(PROVIDER_ENV) or "auto").lower()
    if name == "highlightly":
        return hl.Client()
    if name in ("api_football", "api-football"):
        return af.Client()
    h = hl.Client()
    if h.configured:
        return h
    a = af.Client()
    return a if a.configured else h


def provider_errors(provider) -> tuple:
    """The exception types that mean "not configured" for this provider."""
    return (af.NotConfigured, hl.NotConfigured)

# Cases fuzzy matching gets wrong or would rank ambiguously.
ALIASES = {
    "manchester united": "Man United",
    "manchester city": "Man City",
    "nottingham forest": "Nott'm Forest",
    "newcastle united": "Newcastle",
    "wolverhampton wanderers": "Wolves",
    "brighton and hove albion": "Brighton",
    "west ham united": "West Ham",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "sheffield united": "Sheffield United",
    "ipswich town": "Ipswich",
    "hull city": "Hull",
    "norwich city": "Norwich",
    "cardiff city": "Cardiff",
    "stoke city": "Stoke",
    "swansea city": "Swansea",
    "birmingham city": "Birmingham",
    "queens park rangers": "QPR",
    "west bromwich albion": "West Brom",
    "tottenham hotspur": "Tottenham",
    "afc bournemouth": "Bournemouth",
    "luton town": "Luton",
}

# Lineups are published shortly before kickoff, so there is no point looking
# earlier and no point paying for a fixture already under way.
LOOKAHEAD_HOURS = 6
MIN_QUOTA_HEADROOM = 5


def _norm(name: str) -> str:
    n = name.lower().replace("&", "and")
    for junk in (" football club", " fc", " afc"):
        n = n.replace(junk, "")
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    return " ".join(n.split())


def resolve_team(raw: str, known: list[str]) -> tuple[str | None, str]:
    """Map an external team name onto one of ours.

    Returns (matched_name, how). `how` is one of alias / exact / fuzzy / unmatched,
    so the caller can log how confident the match was.
    """
    n = _norm(raw)
    if n in ALIASES and ALIASES[n] in known:
        return ALIASES[n], "alias"
    by_norm = {_norm(k): k for k in known}
    if n in by_norm:
        return by_norm[n], "exact"
    close = difflib.get_close_matches(n, list(by_norm), n=1, cutoff=0.82)
    if close:
        return by_norm[close[0]], "fuzzy"
    return None, "unmatched"


def _record_unmatched(conn, sport_id: str, source: str, raw: str, context: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO unmatched_names (sport_id, source, raw_name, context, first_seen) "
        "VALUES (?,?,?,?,?)", (sport_id, source, raw, context, now_iso()))


def sync_fixture_ids(client=None, sport_id: str = "pl", season: int | None = None) -> int:
    """Learn API-Football's fixture id for each of our scheduled games.

    One request. Matching is on date plus both resolved team names, so a fixture
    that moved date simply fails to match rather than binding to the wrong game.
    """
    client = client or get_provider()
    if not client.configured:
        raise af.NotConfigured("no lineup provider configured")

    source = client.name
    with connect() as conn:
        with IngestRun(conn, job="ingest.lineups", sport_id=sport_id) as run:
            known = [r["name"] for r in conn.execute(
                "SELECT name FROM teams WHERE sport_id=?", (sport_id,))]
            rows = conn.execute(
                "SELECT g.game_id, g.date_utc, th.name home, ta.name away, g.external_ids_json "
                "FROM games g JOIN teams th ON th.team_id=g.home_team_id "
                "JOIN teams ta ON ta.team_id=g.away_team_id "
                "WHERE g.sport_id=? AND g.status='scheduled'", (sport_id,)).fetchall()
            if not rows:
                log.info("no scheduled games to map")
                return 0

            dates = sorted(r["date_utc"] for r in rows)
            if season is None:
                # API-Football labels a season by the calendar year it starts in.
                y = int(dates[0][:4])
                season = y if int(dates[0][5:7]) >= 7 else y - 1

            try:
                fixtures = client.fixtures_for_dates(dates, season)
            except af.PlanRestriction as exc:
                # Not a fault: the key works, the plan does not reach this season.
                set_state(conn, PLAN_STATE_KEY, str(exc))
                log.warning("lineups unavailable on this plan: %s", exc)
                return 0
            set_state(conn, PLAN_STATE_KEY, None)
            log.info("%d fixtures returned for season %s via %s",
                     len(fixtures), season, source)

            index = {}
            for f in fixtures:
                h, how_h = resolve_team(f["home"], known)
                a, how_a = resolve_team(f["away"], known)
                if h is None:
                    _record_unmatched(conn, sport_id, source, f["home"], "fixture home")
                if a is None:
                    _record_unmatched(conn, sport_id, source, f["away"], "fixture away")
                if h and a:
                    index[(f["date"], h, a)] = f
                    if "fuzzy" in (how_h, how_a):
                        log.info("fuzzy team match: %s -> %s, %s -> %s",
                                 f["home"], h, f["away"], a)

            for r in rows:
                f = index.get((r["date_utc"], r["home"], r["away"]))
                if not f:
                    continue
                ext = json.loads(r["external_ids_json"] or "{}")
                if ext.get(source) == f["fixture_id"]:
                    continue
                ext[source] = f["fixture_id"]
                conn.execute("UPDATE games SET external_ids_json=? WHERE game_id=?",
                             (json.dumps(ext), r["game_id"]))
                run.rows += 1
            log.info("mapped %d fixture ids (%d api calls)", run.rows, client.calls)
            return run.rows


def fetch_lineups(client=None, sport_id: str = "pl",
                  lookahead_hours: int = LOOKAHEAD_HOURS, max_calls: int = 20,
                  now: datetime | None = None) -> int:
    """Fetch lineups for imminent fixtures we do not already have."""
    client = client or get_provider()
    if not client.configured:
        raise af.NotConfigured("no lineup provider configured")
    source = client.name
    now = now or datetime.now(timezone.utc)
    horizon = now + timedelta(hours=lookahead_hours)

    with connect() as conn:
        with IngestRun(conn, job="ingest.lineups", sport_id=sport_id) as run:
            known = [r["name"] for r in conn.execute(
                "SELECT name FROM teams WHERE sport_id=?", (sport_id,))]
            ids = {r["name"]: r["team_id"] for r in conn.execute(
                "SELECT team_id, name FROM teams WHERE sport_id=?", (sport_id,))}
            candidates = conn.execute(
                "SELECT g.game_id, g.date_utc, g.kickoff, g.external_ids_json "
                "FROM games g WHERE g.sport_id=? AND g.status='scheduled' "
                "AND g.external_ids_json LIKE ? "
                "AND NOT EXISTS (SELECT 1 FROM lineups l WHERE l.game_id=g.game_id) "
                "ORDER BY g.date_utc, g.kickoff",
                (sport_id, f'%"{source}"%')).fetchall()

            due = []
            for r in candidates:
                match_day = datetime.fromisoformat(
                    f"{r['date_utc']}T00:00:00+00:00").date()
                if r["kickoff"]:
                    try:
                        when = datetime.fromisoformat(
                            f"{r['date_utc']}T{r['kickoff']}:00+00:00")
                    except ValueError:
                        when = None
                else:
                    when = None
                if when is not None:
                    # Known kickoff: a tight window around it.
                    if now - timedelta(hours=3) <= when <= horizon:
                        due.append(r)
                elif now.date() <= match_day <= horizon.date():
                    # Unknown kickoff. Treating it as midnight would make every
                    # such fixture look long finished, so the whole match day is
                    # eligible instead.
                    due.append(r)
            if not due:
                log.info("no fixtures within %dh awaiting lineups", lookahead_hours)
                return 0

            for r in due[:max_calls]:
                fixture_id = json.loads(r["external_ids_json"])[source]
                try:
                    payload = client.lineups(fixture_id)
                except af.PlanRestriction as exc:
                    set_state(conn, PLAN_STATE_KEY, str(exc))
                    log.warning("lineups unavailable on this plan: %s", exc)
                    break
                except (af.ApiFootballError, hl.HighlightlyError) as exc:
                    log.warning("lineup fetch failed for %s: %s", r["game_id"], exc)
                    break            # quota or outage; stop rather than burn calls
                # A provider may answer with a well-formed record whose XI is
                # empty because the lineup is not out yet. Storing that would
                # satisfy the "already have it" check forever, so it is treated
                # as absent and retried on a later run.
                usable = [ln for ln in payload if ln.get("start_xi")]
                if not usable:
                    log.info("lineups not published yet for %s", r["game_id"])
                    continue
                payload = usable
                for ln in payload:
                    team, _ = resolve_team(ln["team_name"], known)
                    if team is None or team not in ids:
                        _record_unmatched(conn, sport_id, source, ln["team_name"], "lineup")
                        continue
                    conn.execute(
                        "INSERT OR REPLACE INTO lineups (game_id, team_id, formation, coach,"
                        " players_json, source, fetched_at) VALUES (?,?,?,?,?,?,?)",
                        (r["game_id"], ids[team], ln.get("formation"), ln.get("coach"),
                         json.dumps({"start_xi": ln.get("start_xi", []),
                                     "substitutes": ln.get("substitutes", [])}),
                         source, now_iso()))
                    run.rows += 1
            log.info("stored %d lineups (%d api calls)", run.rows, client.calls)
            return run.rows


def plan_restriction() -> str | None:
    """The provider's own explanation, if the plan cannot serve current fixtures."""
    return get_state(PLAN_STATE_KEY)


def for_game(game_id: str) -> dict | None:
    """Both teams' lineups for one game, keyed home/away."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT l.*, t.name, g.home_team_id FROM lineups l "
            "JOIN teams t ON t.team_id=l.team_id JOIN games g ON g.game_id=l.game_id "
            "WHERE l.game_id=?", (game_id,)).fetchall()
    if not rows:
        return None
    out = {}
    for r in rows:
        side = "home" if r["team_id"] == r["home_team_id"] else "away"
        out[side] = {"team": r["name"], "formation": r["formation"], "coach": r["coach"],
                     "source": r["source"], "fetched_at": r["fetched_at"],
                     **json.loads(r["players_json"])}
    return out or None


def unmapped_count(sport_id: str = "pl", source: str | None = None) -> int:
    """Scheduled games with no fixture id for the active provider."""
    source = source or get_provider().name
    with connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) c FROM games WHERE sport_id=? AND status='scheduled' "
            "AND external_ids_json NOT LIKE ?", (sport_id, f'%"{source}"%')).fetchone()["c"]
