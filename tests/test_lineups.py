"""Lineup ingestion, tested against canned API-Football payloads.

No network. The shapes below mirror the documented v3 responses; if the provider
changes them the parsers return None rather than raising, which these tests pin.
"""
import json
from datetime import datetime, timezone

import pytest

from hypz import db, lineups
from hypz.adapters import api_football as af
from hypz.adapters.base import GameRecord, SportConfig
from hypz.ingest import ingest_results

CFG = SportConfig(sport_id="pl", name="PL", simulation_unit="match_goals",
                  extra={"matches_per_season": 380})


class FakeAdapter:
    config = CFG

    def __init__(self, recs):
        self.recs = recs

    def fetch_results(self, seasons=None):
        return self.recs

    def seasons_after(self, y):
        return []

    def current_season_year(self):
        return 2026


FIXTURE_PAYLOAD = [{
    "fixture": {"id": 12345, "date": "2026-08-30T14:00:00+00:00",
                "status": {"short": "NS"}},
    "teams": {"home": {"id": 33, "name": "Manchester United"},
              "away": {"id": 40, "name": "Nottingham Forest"}},
}]

LINEUP_PAYLOAD = [
    {"team": {"id": 33, "name": "Manchester United"},
     "coach": {"id": 1, "name": "A Manager"}, "formation": "4-2-3-1",
     "startXI": [{"player": {"id": 1, "name": "Onana", "number": 24, "pos": "G", "grid": "1:1"}},
                 {"player": {"id": 2, "name": "Shaw", "number": 23, "pos": "D", "grid": "2:1"}}],
     "substitutes": [{"player": {"id": 3, "name": "Bayindir", "number": 1, "pos": "G", "grid": None}}]},
    {"team": {"id": 40, "name": "Nottingham Forest"},
     "coach": {"id": 2, "name": "B Manager"}, "formation": "4-3-3",
     "startXI": [{"player": {"id": 4, "name": "Sels", "number": 1, "pos": "G", "grid": "1:1"}}],
     "substitutes": []},
]


class StubClient(af.Client):
    """Same interface, canned answers, counts calls like the real one."""

    def __init__(self, fixtures=None, lineups_payload=None, key="test-key"):
        super().__init__(key=key)
        self._fixtures = fixtures if fixtures is not None else FIXTURE_PAYLOAD
        self._lineups = lineups_payload if lineups_payload is not None else LINEUP_PAYLOAD

    def fixtures(self, season, league=af.PREMIER_LEAGUE, date_from=None, date_to=None):
        self.calls += 1
        return self._fixtures

    def lineups(self, fixture_id):
        self.calls += 1
        return self._lineups


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.delenv(af.KEY_ENV, raising=False)
    db.init_db()


def _seed_fixture():
    """One scheduled game matching the canned fixture payload."""
    ingest_results(FakeAdapter([
        GameRecord(season="2026/27", date_utc="2026-08-30",
                   home_team="Man United", away_team="Nott'm Forest"),
    ]), force=True)


# ---------------------------------------------------------------- name resolution
KNOWN = ["Man United", "Nott'm Forest", "Brighton", "Wolves", "West Ham", "Tottenham"]


@pytest.mark.parametrize("raw,expected,how", [
    ("Manchester United", "Man United", "alias"),
    ("Nottingham Forest", "Nott'm Forest", "alias"),
    ("Wolverhampton Wanderers", "Wolves", "alias"),
    ("Brighton & Hove Albion", "Brighton", "alias"),
    ("Tottenham", "Tottenham", "exact"),
    ("Brighton FC", "Brighton", "exact"),
])
def test_resolve_known_names(raw, expected, how):
    assert lineups.resolve_team(raw, KNOWN) == (expected, how)


def test_resolve_returns_unmatched_rather_than_guessing():
    name, how = lineups.resolve_team("Real Madrid", KNOWN)
    assert name is None and how == "unmatched"


# ---------------------------------------------------------------- parsers
def test_parse_fixture():
    f = af.parse_fixture(FIXTURE_PAYLOAD[0])
    assert f["fixture_id"] == 12345
    assert f["date"] == "2026-08-30"
    assert (f["home"], f["away"]) == ("Manchester United", "Nottingham Forest")


def test_parse_lineup():
    ln = af.parse_lineup(LINEUP_PAYLOAD[0])
    assert ln["formation"] == "4-2-3-1"
    assert ln["coach"] == "A Manager"
    assert [p["name"] for p in ln["start_xi"]] == ["Onana", "Shaw"]
    assert ln["substitutes"][0]["number"] == 1


@pytest.mark.parametrize("bad", [{}, {"fixture": {}}, {"team": {"id": "x"}}, None])
def test_parsers_return_none_on_unrecognised_shapes(bad):
    """A single odd record must not abort the batch."""
    assert af.parse_fixture(bad if isinstance(bad, dict) else {}) is None
    assert af.parse_lineup(bad if isinstance(bad, dict) else {}) is None


# ---------------------------------------------------------------- configuration
def test_unconfigured_client_raises_not_configured():
    c = af.Client(key="")
    assert not c.configured
    with pytest.raises(af.NotConfigured):
        c.fixtures(2026)


def test_limitations_mention_lineups_only_when_unconfigured():
    from hypz import matchcard
    assert "lineups" in matchcard.limitations()
    assert "live" in matchcard.limitations()


# ---------------------------------------------------------------- ingestion flow
def test_sync_maps_fixture_ids():
    _seed_fixture()
    c = StubClient()
    assert lineups.sync_fixture_ids(c, "pl") == 1
    with db.connect() as conn:
        ext = json.loads(conn.execute(
            "SELECT external_ids_json FROM games").fetchone()["external_ids_json"])
    assert ext[lineups.SOURCE] == 12345
    assert c.calls == 1, "fixture sync must cost exactly one request"


def test_fetch_uses_kickoff_window_when_time_is_known():
    ingest_results(FakeAdapter([
        GameRecord(season="2026/27", date_utc="2026-08-30",
                   home_team="Man United", away_team="Nott'm Forest",
                   extra={"kickoff": "14:00"}),
    ]), force=True)
    c = StubClient()
    lineups.sync_fixture_ids(c, "pl")
    # Three hours before kickoff: inside the window.
    assert lineups.fetch_lineups(
        c, "pl", now=datetime(2026, 8, 30, 11, 0, tzinfo=timezone.utc)) == 2


def test_unknown_kickoff_makes_the_whole_match_day_eligible():
    """Defaulting an unknown kickoff to midnight made such fixtures look finished
    twelve hours early, so their lineups were never fetched."""
    _seed_fixture()          # seeded without a kickoff time
    c = StubClient()
    lineups.sync_fixture_ids(c, "pl")
    assert lineups.fetch_lineups(
        c, "pl", now=datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)) == 2


def test_fetch_stores_both_lineups():
    _seed_fixture()
    c = StubClient()
    lineups.sync_fixture_ids(c, "pl")
    now = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    assert lineups.fetch_lineups(c, "pl", now=now) == 2

    with db.connect() as conn:
        gid = conn.execute("SELECT game_id FROM games").fetchone()["game_id"]
    got = lineups.for_game(gid)
    assert set(got) == {"home", "away"}
    assert got["home"]["formation"] == "4-2-3-1"
    assert got["home"]["coach"] == "A Manager"
    assert [p["name"] for p in got["home"]["start_xi"]] == ["Onana", "Shaw"]


def test_stored_lineups_are_never_refetched():
    """The free tier is 100 requests a day; paying twice for the same data is the
    easiest way to run out."""
    _seed_fixture()
    c = StubClient()
    lineups.sync_fixture_ids(c, "pl")
    now = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    lineups.fetch_lineups(c, "pl", now=now)
    before = c.calls
    assert lineups.fetch_lineups(c, "pl", now=now) == 0
    assert c.calls == before, "refetched a lineup already stored"


def test_fixtures_outside_the_window_cost_nothing():
    _seed_fixture()
    c = StubClient()
    lineups.sync_fixture_ids(c, "pl")
    before = c.calls
    early = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)   # five days out
    assert lineups.fetch_lineups(c, "pl", now=early) == 0
    assert c.calls == before


def test_unmatched_team_is_queued_not_guessed():
    _seed_fixture()
    payload = [dict(FIXTURE_PAYLOAD[0])]
    payload[0] = json.loads(json.dumps(FIXTURE_PAYLOAD[0]))
    payload[0]["teams"]["home"]["name"] = "Some Unknown Club"
    lineups.sync_fixture_ids(StubClient(fixtures=payload), "pl")
    with db.connect() as conn:
        rows = conn.execute("SELECT raw_name FROM unmatched_names").fetchall()
    assert [r["raw_name"] for r in rows] == ["Some Unknown Club"]


def test_empty_lineup_response_is_not_an_error():
    """Lineups appear ~an hour before kickoff; asking early returns nothing."""
    _seed_fixture()
    c = StubClient(lineups_payload=[])
    lineups.sync_fixture_ids(c, "pl")
    now = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    assert lineups.fetch_lineups(c, "pl", now=now) == 0


# ---------------------------------------------------------------- plan coverage
class RestrictedClient(StubClient):
    """A valid key on a plan that does not reach the requested season. This is
    what API-Football's free tier actually does for the current season."""

    MESSAGE = "Free plans do not have access to this season, try from 2022 to 2024."

    def fixtures(self, season, league=af.PREMIER_LEAGUE, date_from=None, date_to=None):
        self.calls += 1
        raise af.PlanRestriction(self.MESSAGE)


def test_plan_restriction_is_parsed_from_a_200_response():
    """The API returns HTTP 200 with an `errors` payload, so this must not be
    mistaken for success."""
    import requests

    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"errors": {"plan": RestrictedClient.MESSAGE}, "response": []}

        @staticmethod
        def raise_for_status():
            return None

    c = af.Client(key="k")
    orig = requests.get
    requests.get = lambda *a, **k: R()
    try:
        with pytest.raises(af.PlanRestriction) as exc:
            c.fixtures(2026)
    finally:
        requests.get = orig
    assert "does not have access" in str(exc.value).lower() or "not have access" in str(exc.value)


def test_restriction_is_recorded_not_raised():
    """A plan limit is a configuration fact, not an outage: the run completes and
    the reason is stored so the page can explain the absence."""
    _seed_fixture()
    assert lineups.sync_fixture_ids(RestrictedClient(), "pl") == 0
    assert RestrictedClient.MESSAGE in lineups.plan_restriction()


def test_page_explains_the_restriction_even_though_a_key_exists(monkeypatch):
    """Regression: limitations() checked only whether a key was configured, so
    once a key existed the page stopped explaining why lineups were missing."""
    from hypz import matchcard
    monkeypatch.setenv(af.KEY_ENV, "a-configured-key")
    _seed_fixture()
    lineups.sync_fixture_ids(RestrictedClient(), "pl")
    lim = matchcard.limitations()
    assert "lineups" in lim
    assert "plan" in lim["lineups"].lower()


def test_restriction_clears_once_the_plan_covers_the_season():
    _seed_fixture()
    lineups.sync_fixture_ids(RestrictedClient(), "pl")
    assert lineups.plan_restriction()
    lineups.sync_fixture_ids(StubClient(), "pl")          # a plan that works
    assert lineups.plan_restriction() is None
