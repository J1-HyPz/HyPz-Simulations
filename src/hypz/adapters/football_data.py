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

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FIRST_SEASON_START = 1993

CONFIG = SportConfig(
    sport_id="pl",
    name="English Premier League",
    simulation_unit="match_goals",
    extra={"country": "England", "tier": 1, "teams_per_season": 20},
)


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
    config = CONFIG

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.raw_dir = RAW_DIR / "football-data" / "E0"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def available_seasons(self) -> list[str]:
        return [
            season_label(y)
            for y in range(FIRST_SEASON_START, current_season_start() + 1)
        ]

    def _fetch_raw(self, start_year: int) -> bytes | None:
        code = season_code(start_year)
        resp = requests.get(BASE_URL.format(code=code), timeout=self.timeout)
        if resp.status_code == 404:
            log.warning("season %s not published yet (404)", season_label(start_year))
            return None
        resp.raise_for_status()
        # Raw store first, parse second.
        (self.raw_dir / f"{code}.csv").write_bytes(resp.content)
        return resp.content

    def _parse(self, raw: bytes, start_year: int) -> list[GameRecord]:
        from io import BytesIO

        # latin-1: these files carry the occasional non-UTF8 byte in referee names.
        df = pd.read_csv(
            BytesIO(raw), encoding="latin-1", on_bad_lines="skip", low_memory=False
        )
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{season_label(start_year)} missing columns: {sorted(missing)}")

        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        # Date is dd/mm/yy in early files and dd/mm/yyyy later; 'mixed' handles both.
        dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
        df = df.assign(match_date=dates).dropna(subset=["match_date"])

        label = season_label(start_year)
        records: list[GameRecord] = []
        for row in df.itertuples(index=False):
            extra = {}
            # Closing odds where present - the free market benchmark for Phase 2.
            for src, dst in (("PSCH", "close_h"), ("PSCD", "close_d"), ("PSCA", "close_a")):
                val = getattr(row, src, None)
                if val is not None and pd.notna(val):
                    extra[dst] = float(val)
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

    def fetch_results(self, seasons: list[str] | None = None) -> list[GameRecord]:
        wanted = (
            range(FIRST_SEASON_START, current_season_start() + 1)
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
