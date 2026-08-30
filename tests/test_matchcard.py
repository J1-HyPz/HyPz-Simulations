"""Schedule and match-card context.

The bounding tests exist because of a real bug: Coventry's previous top-flight
match was in 2001, so the first version reported 9,225 days of "rest" and quoted
2001 results as current form.
"""
import pytest

from hypz import db, matchcard
from hypz.adapters import api_football as af
from hypz.adapters.base import GameRecord, SportConfig
from hypz.ingest import ingest_results

CFG = SportConfig(sport_id="tst", name="T", simulation_unit="match_goals",
                  extra={"matches_per_season": 4})


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


def g(date, home, away, hs, aa, season="2026/27"):
    return GameRecord(season=season, date_utc=date, home_team=home, away_team=away,
                      home_score=hs, away_score=aa)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    # Isolate from the host: whether a real key happens to be exported must not
    # decide whether these pass.
    monkeypatch.delenv(af.KEY_ENV, raising=False)
    monkeypatch.delenv("HIGHLIGHTLY_KEY", raising=False)
    monkeypatch.delenv("HYPZ_LINEUP_PROVIDER", raising=False)
    db.init_db()


def _seed(recs):
    ingest_results(FakeAdapter(recs), force=True)


def test_form_reads_oldest_to_newest():
    _seed([g("2026-08-01", "A", "B", 2, 0), g("2026-08-08", "A", "C", 0, 1),
           g("2026-08-15", "A", "D", 1, 1)])
    with db.connect() as conn:
        recent = matchcard._recent(conn, "tst", "A", "2026-08-20")
    assert matchcard._form_summary(recent)["form"] == "WLD"


def test_form_ignores_matches_older_than_the_window():
    """A promoted side's previous spell is not current form."""
    _seed([g("2001-05-01", "Old", "B", 3, 0, season="2000/01"),
           g("2026-08-15", "A", "B", 1, 0)])
    with db.connect() as conn:
        recent = matchcard._recent(conn, "tst", "Old", "2026-08-30")
    summary = matchcard._form_summary(recent)
    assert summary["played"] == 0
    assert summary["form"] == ""
    assert "no recent matches" in summary["note"]


def test_rest_days_within_the_window():
    _seed([g("2026-08-24", "A", "B", 1, 0)])
    with db.connect() as conn:
        assert matchcard._rest_days(conn, "tst", "A", "2026-08-30") == 6


def test_rest_days_none_across_a_long_gap():
    """25 years between top-flight matches is not rest."""
    _seed([g("2001-05-01", "Old", "B", 1, 0, season="2000/01")])
    with db.connect() as conn:
        assert matchcard._rest_days(conn, "tst", "Old", "2026-08-30") is None


def test_head_to_head_counts_from_the_home_perspective():
    """Meetings at either venue count, but wins are attributed to the side that
    will be at home in the upcoming fixture."""
    _seed([g("2026-08-01", "A", "B", 2, 0),    # A win, A at home
           g("2026-08-08", "B", "A", 3, 1),    # B win, A away
           g("2026-08-15", "B", "A", 1, 1)])   # draw
    with db.connect() as conn:
        h2h = matchcard._head_to_head(conn, "tst", "A", "B", "2026-08-30")
    assert (h2h["home_wins"], h2h["draws"], h2h["away_wins"]) == (1, 1, 1)
    assert len(h2h["meetings"]) == 3


def test_schedule_window_is_inclusive_and_bounded():
    _seed([g("2026-08-29", "A", "B", 1, 0), g("2026-09-02", "C", "D", 0, 0),
           g("2026-09-30", "A", "C", 2, 2)])
    s = matchcard.schedule("tst", days=7, today="2026-08-30", back=0)
    dates = [d["date"] for d in s["days"]]
    assert "2026-09-02" in dates
    assert "2026-08-29" not in dates, "included a fixture before the start date"
    assert "2026-09-30" not in dates, "included a fixture beyond the window"


def test_schedule_lookback_includes_recent_results():
    """Recent results belong in the schedule, not a duplicate panel."""
    _seed([g("2026-08-24", "A", "B", 2, 1), g("2026-09-02", "C", "D", 0, 0)])
    s = matchcard.schedule("tst", days=7, today="2026-08-30", back=7)
    dates = [d["date"] for d in s["days"]]
    assert dates == ["2026-08-24", "2026-09-02"]
    assert s["today"] == "2026-08-30"


def test_card_reports_what_the_source_cannot_provide():
    """With no lineup provider configured, both gaps are named explicitly."""
    _seed([g("2026-08-29", "A", "B", 1, 0)])
    s = matchcard.schedule("tst", days=7, today="2026-08-29", back=0)
    card = s["days"][0]["games"][0]
    assert set(card["unavailable"]) == {"lineups", "live"}
    assert card["result"]["summary"].startswith("A beat B 1")
