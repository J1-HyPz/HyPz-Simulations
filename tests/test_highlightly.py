"""Highlightly provider.

Chosen because its free tier serves the current season; API-Football's does not.
The response shape differs in one structurally important way: the starting XI
arrives nested one row per formation line rather than flat.
"""
import pytest

from hypz.adapters import highlightly as hl

# Shape per the published documentation.
MATCHES = {"data": [{
    "id": 99001,
    "date": "2026-08-31T19:00:00Z",
    "state": {"description": "Not started", "clock": None, "score": None},
    "homeTeam": {"id": 7, "name": "Aston Villa"},
    "awayTeam": {"id": 1, "name": "Arsenal"},
    "league": {"id": 33, "name": "Premier League"},
}]}

LINEUPS = {
    "homeTeam": {
        "id": 7, "name": "Aston Villa", "formation": "4-4-2",
        "initialLineup": [
            [{"name": "E. Martinez", "number": 1, "position": "G"}],
            [{"name": "M. Cash", "number": 2, "position": "D"},
             {"name": "E. Konsa", "number": 4, "position": "D"}],
        ],
        "substitutes": [{"id": 9, "name": "R. Olsen", "number": 26, "position": "G"}],
    },
    "awayTeam": {
        "id": 1, "name": "Arsenal", "formation": "4-3-3",
        "initialLineup": [[{"name": "D. Raya", "number": 22, "position": "G"}]],
        "substitutes": [],
    },
}


def test_parse_match():
    f = hl.parse_match(MATCHES["data"][0])
    assert f["fixture_id"] == 99001
    assert f["date"] == "2026-08-31"
    assert (f["home"], f["away"]) == ("Aston Villa", "Arsenal")
    assert f["status"] == "NOT STARTED"


def test_nested_starting_xi_is_flattened():
    """initialLineup is rows-per-formation-line, unlike API-Football's flat list.
    Flattening must preserve order, goalkeeper first."""
    ln = hl.parse_team_lineup(LINEUPS["homeTeam"])
    assert ln["formation"] == "4-4-2"
    assert [p["name"] for p in ln["start_xi"]] == ["E. Martinez", "M. Cash", "E. Konsa"]
    assert ln["start_xi"][0]["position"] == "G"
    assert [p["name"] for p in ln["substitutes"]] == ["R. Olsen"]


def test_flattening_tolerates_a_flat_list_too():
    """If the provider ever returns a flat list, degrade rather than raise."""
    node = {"name": "X", "formation": "4-4-2",
            "initialLineup": [{"name": "A", "number": 1, "position": "G"}]}
    assert [p["name"] for p in hl.parse_team_lineup(node)["start_xi"]] == ["A"]


@pytest.mark.parametrize("bad", [{}, {"name": ""}, None, [], "nonsense"])
def test_parsers_return_none_on_unrecognised_shapes(bad):
    assert hl.parse_team_lineup(bad) is None
    assert hl.parse_match(bad) is None


def test_records_unwraps_either_shape():
    assert hl.Client._records({"data": [1, 2]}) == [1, 2]
    assert hl.Client._records([1, 2]) == [1, 2]
    assert hl.Client._records({"unexpected": 1}) == []


def test_unconfigured_client_raises():
    c = hl.Client(key="")
    assert not c.configured
    with pytest.raises(hl.NotConfigured):
        c.fixtures_for_dates(["2026-08-31"], 2026)


class StubHL(hl.Client):
    def __init__(self):
        super().__init__(key="k", league_id=33)
        self.dates_requested = []

    def _get(self, path, **params):
        self.calls += 1
        if path.startswith("lineups/"):
            return LINEUPS
        self.dates_requested.append(params.get("date"))
        return MATCHES


def test_one_request_per_date_because_no_range_is_supported():
    c = StubHL()
    fx = c.fixtures_for_dates(["2026-08-31", "2026-09-01", "2026-08-31"], 2026)
    assert c.dates_requested == ["2026-08-31", "2026-09-01"], "duplicate dates not collapsed"
    assert c.calls == 2
    assert len(fx) == 2


def test_lineups_returns_both_sides_normalised():
    c = StubHL()
    out = c.lineups(99001)
    assert [t["team_name"] for t in out] == ["Aston Villa", "Arsenal"]
    assert out[0]["start_xi"][0]["name"] == "E. Martinez"
    assert out[0]["coach"] is None, "this endpoint carries no coach"


def test_provider_selection_prefers_highlightly(monkeypatch):
    """Highlightly's free tier reaches the current season; API-Football's does not,
    so when both are configured the former wins."""
    from hypz import lineups
    monkeypatch.setenv(hl.KEY_ENV, "h-key")
    monkeypatch.setenv("API_FOOTBALL_KEY", "a-key")
    monkeypatch.delenv(lineups.PROVIDER_ENV, raising=False)
    assert lineups.get_provider().name == "highlightly"


def test_provider_selection_falls_back_and_can_be_forced(monkeypatch):
    from hypz import lineups
    monkeypatch.delenv(hl.KEY_ENV, raising=False)
    monkeypatch.setenv("API_FOOTBALL_KEY", "a-key")
    monkeypatch.delenv(lineups.PROVIDER_ENV, raising=False)
    assert lineups.get_provider().name == "api_football"
    monkeypatch.setenv(lineups.PROVIDER_ENV, "highlightly")
    assert lineups.get_provider().name == "highlightly"


def test_unpublished_formation_becomes_none():
    """Before kickoff the endpoint answers with a well-formed placeholder rather
    than an error: formation "Unknown" and empty arrays."""
    node = {"name": "Aston Villa", "formation": "Unknown",
            "initialLineup": [], "substitutes": []}
    ln = hl.parse_team_lineup(node)
    assert ln["formation"] is None
    assert ln["start_xi"] == []
