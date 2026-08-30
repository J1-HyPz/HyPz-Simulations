"""Ingestion invariants: re-running must be safe, and the watermark must only
skip history that is genuinely complete."""
import pandas as pd
import pytest

from hypz import db
from hypz.adapters.base import GameRecord, SportConfig
from hypz.ingest import _complete_through, game_key, ingest_results, load_matches

CFG = SportConfig(sport_id="tst", name="Test League", simulation_unit="match_goals",
                  extra={"matches_per_season": 4, "teams_per_season": 2})


class FakeAdapter:
    """No network. Records how many times results were fetched."""
    config = CFG

    def __init__(self, records):
        self.records = records
        self.calls = []

    def fetch_results(self, seasons=None):
        self.calls.append(seasons)
        if seasons is None:
            return self.records
        return [r for r in self.records if r.season in seasons]

    def seasons_after(self, year):
        return [s for s in sorted({r.season for r in self.records})
                if int(s.split("/")[0]) > year]

    def current_season_year(self):
        return 2026


def _season(label, n=4):
    return [GameRecord(season=label, date_utc=f"{label[:4]}-08-{10+i:02d}",
                       home_team="H" if i % 2 else "A",
                       away_team="A" if i % 2 else "H",
                       home_score=i % 3, away_score=(i + 1) % 3)
            for i in range(n)]


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()


def test_game_key_is_deterministic():
    r = GameRecord(season="2024/25", date_utc="2024-08-17",
                   home_team="Nott'm Forest", away_team="Man City", home_score=1, away_score=1)
    assert game_key("pl", r) == game_key("pl", r)
    assert game_key("pl", r) == "pl:2024/25:2024-08-17:nott-m-forest:man-city"


def test_ingest_is_idempotent():
    a = FakeAdapter(_season("2023/24"))
    first = ingest_results(a, force=True)
    rows_after_first = len(load_matches("tst"))
    ingest_results(a, force=True)
    rows_after_second = len(load_matches("tst"))
    assert first == 4
    assert rows_after_first == rows_after_second == 4, "re-running duplicated rows"


def test_watermark_covers_seasons_that_have_ended():
    """Completeness is "the season is over", not "the count is full" - the source
    ships some historic seasons short and they would never qualify otherwise."""
    recs = _season("2023/24", 4) + _season("2024/25", 2)   # second season short
    a = FakeAdapter(recs)
    ingest_results(a, force=True)
    with db.connect() as conn:
        assert _complete_through(conn, CFG, 2026) == 2024
        assert db.get_watermark(conn, "ingest.results", "tst") == "2024"


def test_watermark_never_covers_the_current_season():
    a = FakeAdapter(_season("2023/24", 4) + _season("2026/27", 4))
    ingest_results(a, force=True)
    with db.connect() as conn:
        assert _complete_through(conn, CFG, 2026) == 2023, "current season marked done"


def test_watermark_stops_at_a_gap():
    """A hole in the middle of history must keep being refetched."""
    recs = _season("2023/24", 4) + _season("2025/26", 4)   # 2024/25 missing entirely
    a = FakeAdapter(recs)
    ingest_results(a, force=True)
    with db.connect() as conn:
        assert _complete_through(conn, CFG, 2026) == 2023, "watermark jumped over a missing season"


def test_second_run_uses_the_watermark():
    a = FakeAdapter(_season("2023/24", 4) + _season("2024/25", 4))
    ingest_results(a, force=True)
    a.calls.clear()
    ingest_results(a)
    assert a.calls, "adapter was never called"
    assert a.calls[0] is not None, "watermark ignored; refetched everything"
    assert "2023/24" not in a.calls[0], "refetched a season already complete"


def test_ingest_run_is_recorded():
    ingest_results(FakeAdapter(_season("2023/24")), force=True)
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM ingest_runs ORDER BY run_id DESC LIMIT 1").fetchone()
    assert row["job"] == "ingest.results"
    assert row["status"] == "success"
    assert row["rows"] == 4


def test_failure_is_recorded_not_swallowed():
    class Broken(FakeAdapter):
        def fetch_results(self, seasons=None):
            raise RuntimeError("source is down")

    with pytest.raises(RuntimeError):
        ingest_results(Broken([]), force=True)
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM ingest_runs ORDER BY run_id DESC LIMIT 1").fetchone()
    assert row["status"] == "failed"
    assert "source is down" in row["error"]


def test_stale_watermark_format_is_ignored_not_fatal():
    """An earlier build stored a date here. Upgrading must not crash ingestion."""
    a = FakeAdapter(_season("2023/24"))
    ingest_results(a, force=True)
    with db.connect() as conn:
        db.set_watermark(conn, "ingest.results", "tst", "2026-08-24")   # legacy value
    a.calls.clear()
    ingest_results(a)                      # must not raise
    assert a.calls[0] is None, "should fall back to a full fetch"


def test_bom_is_stripped_before_parsing():
    """latin-1 decodes a UTF-8 BOM to mojibake, which corrupts the first column
    name and made the Div filter raise KeyError."""
    from io import BytesIO
    import pandas as pd
    from hypz.adapters.football_data import _strip_bom

    payload = "Div,Date\nE0,01/01/2026\n".encode("utf-8")
    with_bom = b"\xef\xbb\xbf" + payload
    assert _strip_bom(with_bom) == payload
    assert _strip_bom(payload) == payload, "stripping must be idempotent"
    df = pd.read_csv(BytesIO(_strip_bom(with_bom)), encoding="latin-1")
    assert list(df.columns) == ["Div", "Date"]
