"""The contract a lineup provider must satisfy.

Two providers exist and they disagree about almost everything at the wire level:
API-Football takes a date range and returns a flat starting XI; Highlightly takes
one date at a time and returns the XI nested one row per formation line. Both are
normalised to the shapes below so `hypz.lineups` never learns which is in use.
"""
from __future__ import annotations

from typing import Protocol, TypedDict


class Fixture(TypedDict):
    fixture_id: int | str
    date: str            # YYYY-MM-DD
    kickoff_utc: str
    status: str
    home: str
    away: str


class Player(TypedDict, total=False):
    name: str
    number: int | None
    position: str | None


class TeamLineup(TypedDict, total=False):
    team_name: str
    formation: str | None
    coach: str | None
    start_xi: list[Player]
    substitutes: list[Player]


class LineupProvider(Protocol):
    name: str
    calls: int

    @property
    def configured(self) -> bool:
        """False when no key is present. Callers skip rather than fail."""

    def fixtures_for_dates(self, dates: list[str], season: int) -> list[Fixture]:
        """Fixtures on the given dates. Implementations decide how many requests
        that costs; `calls` reports the true number."""

    def lineups(self, fixture_id: int | str) -> list[TeamLineup]:
        """Both teams' lineups, or an empty list if not published yet."""

    def quota(self) -> dict:
        """Remaining daily allowance, best effort."""
