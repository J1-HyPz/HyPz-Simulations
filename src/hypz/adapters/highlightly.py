"""Highlightly lineup provider.

Chosen over API-Football because its tiers differ only in request volume; every
tier, including the free one, serves lineups for the current season. API-Football
gates the current season behind a paid plan regardless of volume.

Free tier is 100 requests a day. `/matches` accepts one date at a time rather than
a range, so a matchweek spanning three days costs three requests plus one per
fixture - roughly thirteen.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

from .lineup_provider import Fixture, TeamLineup

log = logging.getLogger(__name__)

BASE = os.environ.get("HIGHLIGHTLY_BASE", "https://soccer.highlightly.net")
KEY_ENV = "HIGHLIGHTLY_KEY"
LEAGUE_ID_ENV = "HIGHLIGHTLY_LEAGUE_ID"
TIMEOUT = 20


class NotConfigured(RuntimeError):
    pass


class HighlightlyError(RuntimeError):
    pass


def _players(rows: Any) -> list[dict]:
    """Flatten Highlightly's lineup rows.

    `initialLineup` is a list of rows, one per formation line, goalkeeper first -
    unlike API-Football's flat list. A defensive isinstance check keeps a shape
    change from raising rather than degrading.
    """
    out: list[dict] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        entries = row if isinstance(row, list) else [row]
        for p in entries:
            if not isinstance(p, dict) or not p.get("name"):
                continue
            out.append({"name": str(p["name"]), "number": p.get("number"),
                        "position": p.get("position")})
    return out


# Before a lineup is published the endpoint returns a well-formed record with
# empty arrays and this literal formation, rather than 404 or an empty body.
UNPUBLISHED_FORMATION = "Unknown"


def parse_team_lineup(node: Any) -> TeamLineup | None:
    if not isinstance(node, dict) or not node.get("name"):
        return None
    formation = node.get("formation")
    if formation == UNPUBLISHED_FORMATION:
        formation = None
    return {
        "team_name": str(node["name"]).strip(),
        "formation": formation,
        "coach": None,          # not provided by this endpoint
        "start_xi": _players(node.get("initialLineup")),
        "substitutes": _players(node.get("substitutes")),
    }


def parse_match(item: Any) -> Fixture | None:
    try:
        home, away = item["homeTeam"], item["awayTeam"]
        state = item.get("state") or {}
        return {
            "fixture_id": item["id"],
            "date": str(item["date"])[:10],
            "kickoff_utc": str(item["date"]),
            "status": str(state.get("description") or "").upper(),
            "home": str(home["name"]).strip(),
            "away": str(away["name"]).strip(),
        }
    except (KeyError, TypeError, ValueError):
        log.warning("skipping unrecognised match record")
        return None


class Client:
    name = "highlightly"

    def __init__(self, key: str | None = None, base: str = BASE,
                 league_id: int | str | None = None):
        # `key=""` means "explicitly unconfigured"; only None falls back to the
        # environment. Without this, a test cannot construct an unconfigured
        # client on a host that happens to export a key.
        self.key = (os.environ.get(KEY_ENV, "") if key is None else key) or ""
        self.base = base.rstrip("/")
        self.calls = 0
        self._league_id = league_id or os.environ.get(LEAGUE_ID_ENV) or None

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def _get(self, path: str, **params) -> Any:
        if not self.configured:
            raise NotConfigured(f"{KEY_ENV} is not set")
        resp = requests.get(f"{self.base}/{path.lstrip('/')}",
                            headers={"x-rapidapi-key": self.key},
                            params={k: v for k, v in params.items() if v is not None},
                            timeout=TIMEOUT)
        self.calls += 1
        if resp.status_code == 429:
            raise HighlightlyError("rate limited (429) - daily quota likely exhausted")
        if resp.status_code in (401, 403):
            raise HighlightlyError(f"rejected ({resp.status_code}) - check {KEY_ENV}")
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _records(body: Any) -> list:
        """Responses come back either bare or wrapped in `data`."""
        if isinstance(body, list):
            return body
        if isinstance(body, dict):
            for key in ("data", "results", "response"):
                if isinstance(body.get(key), list):
                    return body[key]
        return []

    def league_id(self, name: str = "Premier League", country: str = "England") -> int | str:
        if self._league_id:
            return self._league_id
        body = self._get("leagues", leagueName=name, countryName=country, limit=100)
        for rec in self._records(body):
            if not isinstance(rec, dict):
                continue
            if str(rec.get("name", "")).strip().lower() == name.lower():
                self._league_id = rec.get("id")
                log.info("resolved league %r to id %s", name, self._league_id)
                return self._league_id
        raise HighlightlyError(
            f"could not resolve league {name!r}; set {LEAGUE_ID_ENV} explicitly")

    def fixtures_for_dates(self, dates: list[str], season: int) -> list[Fixture]:
        """One request per date - this endpoint takes no range."""
        league = self.league_id()
        out: list[Fixture] = []
        for d in sorted(set(dates)):
            body = self._get("matches", leagueId=league, date=d, season=season, limit=100)
            for rec in self._records(body):
                f = parse_match(rec)
                if f:
                    out.append(f)
        return out

    def lineups(self, fixture_id: int | str) -> list[TeamLineup]:
        body = self._get(f"lineups/{fixture_id}")
        if isinstance(body, list):
            body = body[0] if body else {}
        if not isinstance(body, dict):
            return []
        out = []
        for side in ("homeTeam", "awayTeam"):
            parsed = parse_team_lineup(body.get(side))
            if parsed:
                out.append(parsed)
        return out

    def quota(self) -> dict:
        # No status endpoint is documented; report what we spent this process.
        return {"used": self.calls, "limit": None, "plan": "unknown"}
