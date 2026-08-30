"""API-Football client, used only for what football-data.co.uk cannot supply.

Scope is deliberately narrow: fixtures (to learn their fixture ids) and starting
lineups. Results, statistics and odds keep coming from the free CSV feed, which
has no rate limit and no key.

The free tier allows 100 requests a day, so every call is budgeted: the fixture
list is fetched once a day, and a lineup is fetched at most once per fixture and
never again after it is stored. A typical Premier League matchweek costs about
eleven requests.

The key is read from the environment and never logged.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

log = logging.getLogger(__name__)

BASE = os.environ.get("API_FOOTBALL_BASE", "https://v3.football.api-sports.io")
KEY_ENV = "API_FOOTBALL_KEY"
PREMIER_LEAGUE = 39           # API-Football league id
TIMEOUT = 20


class NotConfigured(RuntimeError):
    """No API key present. Every caller treats this as "skip", not "fail"."""


class ApiFootballError(RuntimeError):
    pass


class PlanRestriction(ApiFootballError):
    """The key is valid but the plan does not cover what was asked for.

    Distinct from a failure: nothing is broken and retrying will not help, so
    callers record it and stop rather than treating it as an outage. These
    responses do not count against the daily quota.
    """


class Client:
    name = "api_football"

    def __init__(self, key: str | None = None, base: str = BASE):
        # `key=""` means "explicitly unconfigured"; only None falls back to the
        # environment. Without this, a test cannot construct an unconfigured
        # client on a host that happens to export a key.
        self.key = (os.environ.get(KEY_ENV, "") if key is None else key) or ""
        self.base = base.rstrip("/")
        self.calls = 0

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def _get(self, path: str, **params) -> list[dict]:
        if not self.configured:
            raise NotConfigured(
                f"{KEY_ENV} is not set; lineups are skipped. See README.")
        url = f"{self.base}/{path.lstrip('/')}"
        resp = requests.get(url, headers={"x-apisports-key": self.key},
                            params=params, timeout=TIMEOUT)
        self.calls += 1
        if resp.status_code == 429:
            raise ApiFootballError("rate limited (429) - daily quota likely exhausted")
        resp.raise_for_status()
        body = resp.json()
        # The API returns 200 with an `errors` payload rather than an HTTP error.
        errors = body.get("errors")
        if errors:
            text = str(errors)
            if isinstance(errors, dict) and "plan" in errors:
                raise PlanRestriction(str(errors["plan"]))
            raise ApiFootballError(f"api error: {text}")
        out = body.get("response", [])
        if not isinstance(out, list):
            raise ApiFootballError(f"unexpected response shape for {path}")
        return out

    def quota(self) -> dict[str, Any]:
        """Remaining daily allowance, so a caller can stop before being cut off."""
        if not self.configured:
            raise NotConfigured(f"{KEY_ENV} is not set")
        resp = requests.get(f"{self.base}/status",
                            headers={"x-apisports-key": self.key}, timeout=TIMEOUT)
        resp.raise_for_status()
        r = resp.json().get("response", {}) or {}
        req = r.get("requests", {}) or {}
        return {"used": req.get("current"), "limit": req.get("limit_day"),
                "plan": (r.get("subscription") or {}).get("plan")}

    def fixtures(self, season: int, league: int = PREMIER_LEAGUE,
                 date_from: str | None = None, date_to: str | None = None) -> list[dict]:
        params: dict[str, Any] = {"league": league, "season": season}
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
        return self._get("fixtures", **params)

    def fixtures_for_dates(self, dates: list[str], season: int) -> list[dict]:
        """This endpoint takes a range, so all dates cost a single request."""
        ds = sorted(set(dates))
        raw = self.fixtures(season, date_from=ds[0], date_to=ds[-1])
        return [f for f in (parse_fixture(i) for i in raw) if f]

    def lineups(self, fixture_id: int) -> list[dict]:
        raw = self._get("fixtures/lineups", fixture=fixture_id)
        return [ln for ln in (parse_lineup(i) for i in raw) if ln]


def parse_fixture(item: dict) -> dict | None:
    """Flatten one fixture. Returns None rather than raising on a shape we do not
    recognise - a single odd record must not abort the batch."""
    try:
        fx, teams = item["fixture"], item["teams"]
        return {
            "fixture_id": int(fx["id"]),
            # ISO 8601 with offset; the date half is what our keys use.
            "date": str(fx["date"])[:10],
            "kickoff_utc": str(fx["date"]),
            "status": ((fx.get("status") or {}).get("short") or "").upper(),
            "home": str(teams["home"]["name"]).strip(),
            "away": str(teams["away"]["name"]).strip(),
            "home_id": int(teams["home"]["id"]),
            "away_id": int(teams["away"]["id"]),
        }
    except (KeyError, TypeError, ValueError):
        log.warning("skipping unrecognised fixture record")
        return None


def parse_lineup(item: dict) -> dict | None:
    """Flatten one team's lineup."""
    try:
        team = item["team"]

        def players(key):
            out = []
            for p in item.get(key) or []:
                pl = (p or {}).get("player") or {}
                if not pl.get("name"):
                    continue
                out.append({
                    "name": str(pl["name"]),
                    "number": pl.get("number"),
                    "position": pl.get("pos"),
                    "grid": pl.get("grid"),
                })
            return out

        return {
            "team_id_external": int(team["id"]),
            "team_name": str(team["name"]).strip(),
            "formation": item.get("formation"),
            "coach": ((item.get("coach") or {}).get("name")),
            "start_xi": players("startXI"),
            "substitutes": players("substitutes"),
        }
    except (KeyError, TypeError, ValueError):
        log.warning("skipping unrecognised lineup record")
        return None
