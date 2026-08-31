"""Premier League results from football-data.co.uk.

Plain CSV over HTTPS, one file per season back to 1993/94 - no scraping, no API
key, no rate limit. Each file is written to the raw store unmodified before it is
parsed (design doc section 4.5), so a parser fix never requires a refetch.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from ..config import RAW_DIR
from .base import GameRecord, SportAdapter, SportConfig

log = logging.getLogger(__name__)

def looks_like_csv(raw: bytes) -> bool:
    """The server answers 200 with an HTML error page for a season it has not
    published yet, rather than 404. Content has to be checked, not the status."""
    head = raw.lstrip()[:200].lower()
    return not head.startswith(b"<") and b"div," in head


def _strip_bom(raw: bytes) -> bytes:
    """These files are UTF-8-with-BOM but carry stray non-UTF8 bytes in referee
    names, so they are parsed as latin-1. That decodes the BOM to mojibake rather
    than to \ufeff, which silently corrupts the first column name - so the BOM is
    removed at the byte level, before any decoding decision."""
    return raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw


BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Team-level match statistics the results feed carries and we previously dropped.
# Column pairs are (home, away) -> our key.
STAT_COLUMNS = {
    ("HS", "AS"): "shots",
    ("HST", "AST"): "shots_on_target",
    ("HC", "AC"): "corners",
    ("HF", "AF"): "fouls",
    ("HY", "AY"): "yellows",
    ("HR", "AR"): "reds",
}
FIRST_SEASON_START = 1993


def season_code(start_year: int) -> str:
    """1993 -> '9394', 2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """1993 -> '1993/94'."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    """A PL season starting in August is labelled by its opening calendar year."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


class FootballDataAdapter(SportAdapter):
    """One instance per league. The division code is the only thing that varies."""

    def __init__(self, sport_id: str = "pl", timeout: int = 30):
        from ..leagues import get as _league
        self.league = _league(sport_id)
        self.division = self.league.code
        self.config = SportConfig(
            sport_id=self.league.sport_id,
            name=self.league.name,
            simulation_unit="match_goals",
            extra={"country": self.league.country,
                   "division": self.league.code,
                   "teams_per_season": self.league.teams,
                   "matches_per_season": self.league.matches_per_season},
        )
        self.timeout = timeout
        self.raw_dir = RAW_DIR / "football-data" / self.division
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def current_season_year(self) -> int:
        return current_season_start()

    @property
    def first_season(self) -> int:
        return self.league.first_season

    def seasons_after(self, start_year: int) -> list[str]:
        """Seasons strictly after `start_year`, up to and including the current one.

        The current season is always included even when it is also the watermark:
        it keeps gaining matches, so it is never finished.
        """
        first = max(start_year + 1, self.first_season)
        return [season_label(y) for y in range(min(first, current_season_start()),
                                               current_season_start() + 1)]

    def available_seasons(self) -> list[str]:
        return [
            season_label(y)
            for y in range(self.first_season, current_season_start() + 1)
        ]

    def _fetch_raw(self, start_year: int) -> bytes | None:
        code = season_code(start_year)
        resp = requests.get(BASE_URL.format(season=code, div=self.division),
                            timeout=self.timeout)
        if resp.status_code == 404:
            log.warning("season %s not published yet (404)", season_label(start_year))
            return None
        resp.raise_for_status()
        if not looks_like_csv(_strip_bom(resp.content)):
            log.warning("%s %s: not published yet (server returned a page, not a CSV)",
                        self.division, season_label(start_year))
            return None
        # Raw store first, parse second.
        (self.raw_dir / f"{code}.csv").write_bytes(resp.content)
        return resp.content

    def _parse(self, raw: bytes, start_year: int) -> list[GameRecord]:
        from io import BytesIO

        # latin-1: these files carry the occasional non-UTF8 byte in referee names.
        df = pd.read_csv(
            BytesIO(_strip_bom(raw)), encoding="latin-1", on_bad_lines="skip",
            low_memory=False
        )
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        missing = required - set(df.columns)
        if missing:
            # Skip rather than raise: one malformed season file must not abort a
            # backfill spanning three decades and eight leagues.
            log.warning("%s %s: skipping, missing columns %s",
                        self.division, season_label(start_year), sorted(missing))
            return []

        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        # Date is dd/mm/yy in early files and dd/mm/yyyy later; 'mixed' handles both.
        dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
        # copy() first: these files have ~100 columns and assigning into the view
        # raises a fragmentation warning on every season of every league.
        df = df.copy()
        df["match_date"] = dates
        df = df.dropna(subset=["match_date"])

        label = season_label(start_year)
        records: list[GameRecord] = []
        for row in df.itertuples(index=False):
            extra = {}
            # Closing odds where present - the free market benchmark for Phase 2.
            for src, dst in (("PSCH", "close_h"), ("PSCD", "close_d"), ("PSCA", "close_a")):
                val = getattr(row, src, None)
                if val is not None and pd.notna(val):
                    extra[dst] = float(val)

            # Match statistics. Absent in the earliest seasons, so every field is
            # optional and a missing one is simply not recorded.
            stats = {}
            for (hcol, acol), key in STAT_COLUMNS.items():
                hv, av = getattr(row, hcol, None), getattr(row, acol, None)
                if hv is not None and av is not None and pd.notna(hv) and pd.notna(av):
                    stats[key] = {"h": int(hv), "a": int(av)}
            if stats:
                extra["stats"] = stats
            for src, dst in (("HTHG", "h"), ("HTAG", "a")):
                val = getattr(row, src, None)
                if val is not None and pd.notna(val):
                    extra.setdefault("half_time", {})[dst] = int(val)
            # Expected goals appear in recent seasons for most divisions. Stored
            # whenever present so the history accumulates without a re-ingest.
            for src, dst in (("HxG", "h"), ("AxG", "a")):
                val = getattr(row, src, None)
                if val is not None and pd.notna(val):
                    try:
                        extra.setdefault("xg", {})[dst] = float(val)
                    except (TypeError, ValueError):
                        pass
            ref = getattr(row, "Referee", None)
            if ref is not None and pd.notna(ref) and str(ref).strip():
                extra["referee"] = str(ref).strip()
            ko = getattr(row, "Time", None)
            if ko is not None and pd.notna(ko) and str(ko).strip():
                extra["kickoff"] = str(ko).strip()
            records.append(
                GameRecord(
                    season=label,
                    date_utc=row.match_date.date().isoformat(),
                    home_team=str(row.HomeTeam).strip(),
                    away_team=str(row.AwayTeam).strip(),
                    home_score=int(row.FTHG),
                    away_score=int(row.FTAG),
                    extra=extra,
                )
            )
        return records

    def fetch_fixtures(self) -> list[GameRecord]:
        """Upcoming fixtures from the all-competitions feed, filtered to the Premier League.

        Keys are built exactly as for results, so when a fixture is later played the
        result upsert lands on the same row rather than creating a duplicate.
        """
        from io import BytesIO

        resp = requests.get(FIXTURES_URL, timeout=self.timeout)
        resp.raise_for_status()
        (self.raw_dir / "fixtures.csv").write_bytes(resp.content)

        df = pd.read_csv(BytesIO(_strip_bom(resp.content)), encoding="latin-1",
                         on_bad_lines="skip", low_memory=False)
        df = df[df["Div"] == self.division].dropna(subset=["HomeTeam", "AwayTeam", "Date"])
        dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
        df = df.assign(match_date=dates).dropna(subset=["match_date"])

        out: list[GameRecord] = []
        for row in df.itertuples(index=False):
            d = row.match_date.date()
            extra = {}
            if getattr(row, "Time", None) is not None and pd.notna(row.Time):
                extra["kickoff"] = str(row.Time).strip()
            ref = getattr(row, "Referee", None)
            if ref is not None and pd.notna(ref) and str(ref).strip():
                extra["referee"] = str(ref).strip()
            # Pre-match prices; the closing line only exists once a match is played.
            for src, dst in (("B365H", "open_h"), ("B365D", "open_d"), ("B365A", "open_a")):
                val = getattr(row, src, None)
                if val is not None and pd.notna(val):
                    extra[dst] = float(val)
            out.append(GameRecord(
                season=season_label(current_season_start(d)),
                date_utc=d.isoformat(),
                home_team=str(row.HomeTeam).strip(),
                away_team=str(row.AwayTeam).strip(),
                extra=extra,
            ))
        log.info("%d upcoming %s fixtures", len(out), self.division)
        return out

    def fetch_results(self, seasons: list[str] | None = None) -> list[GameRecord]:
        wanted = (
            range(self.first_season, current_season_start() + 1)
            if seasons is None
            else [int(s.split("/")[0]) for s in seasons]
        )
        out: list[GameRecord] = []
        for year in wanted:
            raw = self._fetch_raw(year)
            if raw is None:
                continue
            recs = self._parse(raw, year)
            log.info("%s: %d matches", season_label(year), len(recs))
            out.extend(recs)
        return out
