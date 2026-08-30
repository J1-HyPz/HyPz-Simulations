"""Pipeline health from `ingest_runs` (design doc section 4.4 rule 3).

The point of the run table is that a stale scraper is visible rather than silent,
which only works if something actually reads it. This is that something.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .db import connect

# How long each job may go without a successful run before it is considered stale.
STALENESS = {
    "ingest.results": timedelta(hours=36),
    "ingest.fixtures": timedelta(hours=36),
    "model.forecast": timedelta(hours=36),
    "export.web": timedelta(hours=36),
}
DEFAULT_STALENESS = timedelta(hours=48)


def summary() -> list[dict]:
    now = datetime.now(timezone.utc)
    with connect() as conn:
        jobs = [r["job"] for r in conn.execute(
            "SELECT DISTINCT job FROM ingest_runs ORDER BY job")]
        out = []
        for job in jobs:
            last = conn.execute(
                "SELECT * FROM ingest_runs WHERE job=? ORDER BY run_id DESC LIMIT 1",
                (job,)).fetchone()
            ok = conn.execute(
                "SELECT * FROM ingest_runs WHERE job=? AND status='success' "
                "ORDER BY run_id DESC LIMIT 1", (job,)).fetchone()
            fails = conn.execute(
                "SELECT COUNT(*) c FROM ingest_runs WHERE job=? AND status='failed'",
                (job,)).fetchone()["c"]
            age = None
            if ok:
                age = now - datetime.fromisoformat(ok["started_at"])
            limit = STALENESS.get(job, DEFAULT_STALENESS)
            if last["status"] == "failed":
                state = "failing"
            elif age is None or age > limit:
                state = "stale"
            else:
                state = "ok"
            out.append({
                "job": job, "state": state,
                "last_status": last["status"], "last_at": last["started_at"],
                "last_rows": last["rows"], "last_error": last["error"],
                "last_success_at": ok["started_at"] if ok else None,
                "age_hours": round(age.total_seconds() / 3600, 1) if age else None,
                "total_failures": fails,
            })
        return out


def overall(rows: list[dict]) -> str:
    if any(r["state"] == "failing" for r in rows):
        return "failing"
    if any(r["state"] == "stale" for r in rows):
        return "stale"
    return "ok" if rows else "unknown"
