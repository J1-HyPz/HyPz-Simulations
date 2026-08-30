"""Unattended scheduling (design doc section 4.3).

Runs as the container's main process. Every job goes through `_run`, which never
lets an exception escape: one failing source must not take the scheduler down with
it, because a dead scheduler is invisible while a failed run is recorded and shows
up in `hypz health`.

Cadences are deliberately less aggressive than section 4.3. Results appear once a
day, the fixtures feed changes slowly, and the forecast only rewrites rows whose
inputs_hash actually moved - so polling harder buys nothing.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .adapters.football_data import FootballDataAdapter
from .export_web import export
from .ingest import ingest_fixtures, ingest_results
from .predict import forecast_scheduled

log = logging.getLogger(__name__)

WEB_OUT = Path(os.environ.get("HYPZ_WEB_OUT", "/data/web/fixture-model.html"))
TZ = os.environ.get("TZ", "Europe/London")


def _run(name, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        log.info("job %s finished: %s", name, result)
    except Exception:
        # Deliberately swallowed. ingest_runs already recorded the failure, and
        # killing the scheduler would turn one bad fetch into total silence.
        log.exception("job %s failed", name)


def job_results():
    _run("ingest.results", ingest_results, FootballDataAdapter())


def job_fixtures():
    _run("ingest.fixtures", ingest_fixtures, FootballDataAdapter())


def job_forecast():
    _run("model.forecast", forecast_scheduled, "pl")


def job_export():
    _run("export.web", export, WEB_OUT)


def build_scheduler() -> BlockingScheduler:
    sched = BlockingScheduler(timezone=TZ)
    cron = lambda **kw: CronTrigger(timezone=TZ, **kw)
    # Results land overnight; everything downstream follows in order.
    sched.add_job(job_results, cron(hour=4, minute=0), id="ingest.results")
    sched.add_job(job_fixtures, cron(hour="8,20", minute=0), id="ingest.fixtures")
    sched.add_job(job_forecast, cron(hour=5, minute=0), id="model.forecast")
    sched.add_job(job_export, cron(hour=5, minute=15), id="export.web")
    return sched


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    log.info("scheduler starting, timezone %s", TZ)

    if os.environ.get("HYPZ_RUN_ON_START", "1") == "1":
        # A fresh container should be useful immediately rather than at 04:00.
        log.info("priming: running every job once")
        job_results(); job_fixtures(); job_forecast(); job_export()

    sched = build_scheduler()
    for j in sched.get_jobs():
        log.info("scheduled %s -> %s", j.id, j.trigger)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler stopping")


if __name__ == "__main__":
    main()
