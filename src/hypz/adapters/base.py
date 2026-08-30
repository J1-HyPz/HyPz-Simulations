"""The sport adapter contract (design doc section 4.1).

The single most important boundary in the system: the core knows nothing about
any specific sport, so adding one means writing an adapter, not touching the
engine. Phase 1 needs only schedule/results; the rest of the interface arrives
with the sports that require it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SportConfig:
    """Structural parameters of a sport, consumed by the modelling layer."""

    sport_id: str
    name: str
    # "match_goals" (closed-form bivariate Poisson) or "drive" (Monte Carlo).
    simulation_unit: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GameRecord:
    """One fixture, with a result attached if it has been played."""

    season: str
    date_utc: str
    home_team: str
    away_team: str
    home_score: int | None = None
    away_score: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_played(self) -> bool:
        return self.home_score is not None and self.away_score is not None


class SportAdapter(Protocol):
    config: SportConfig

    def fetch_results(self, seasons: list[str] | None = None) -> list[GameRecord]:
        """Return played fixtures. Must be safe to call repeatedly."""
        ...

    def available_seasons(self) -> list[str]:
        ...
