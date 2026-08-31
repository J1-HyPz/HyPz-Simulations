"""The leagues this deployment tracks.

Adding one is a row here, because the sport adapter contract (design doc section
4.1) keeps the core ignorant of any particular competition and every table is
already keyed by sport_id.

`code` is football-data.co.uk's division code. Every league listed carries the
same rich columns as the Premier League - shots, shots on target, corners, fouls,
cards, half-time score and Pinnacle closing odds - so nothing downstream needs to
know which one it is looking at.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    sport_id: str
    code: str            # football-data division code
    name: str
    country: str
    teams: int
    first_season: int = 1993

    @property
    def matches_per_season(self) -> int:
        return self.teams * (self.teams - 1)


LEAGUES: dict[str, League] = {
    l.sport_id: l for l in [
        League("pl",           "E0",  "Premier League",  "England",     20),
        League("laliga",       "SP1", "La Liga",         "Spain",       20),
        League("seriea",       "I1",  "Serie A",         "Italy",       20),
        League("bundesliga",   "D1",  "Bundesliga",      "Germany",     18),
        League("ligue1",       "F1",  "Ligue 1",         "France",      18),
        League("eredivisie",   "N1",  "Eredivisie",      "Netherlands", 18),
        League("primeira",     "P1",  "Primeira Liga",   "Portugal",    18),
        League("championship", "E1",  "Championship",    "England",     24),
    ]
}

DEFAULT = "pl"


def get(sport_id: str) -> League:
    try:
        return LEAGUES[sport_id]
    except KeyError:
        raise SystemExit(
            f"unknown league {sport_id!r}; known: {', '.join(sorted(LEAGUES))}") from None


def all_ids() -> list[str]:
    return list(LEAGUES)
