"""Command line entry point for the Phase 1 vertical slice."""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from .adapters.football_data import FootballDataAdapter
from .db import connect, init_db, now_iso
from .ingest import ingest_results, load_matches
from . import backtest as bt
from .export_web import export as export_web
from .models import dixon_coles

ADAPTERS = {"pl": FootballDataAdapter}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_init(args) -> None:
    init_db()
    print("schema initialised")


def cmd_ingest(args) -> None:
    adapter = ADAPTERS[args.sport]()
    seasons = args.seasons.split(",") if args.seasons else None
    n = ingest_results(adapter, seasons)
    print(f"ingested {n} records")


def cmd_status(args) -> None:
    with connect() as conn:
        for sport in conn.execute("SELECT sport_id, name FROM sports"):
            g = conn.execute(
                "SELECT COUNT(*) c FROM games WHERE sport_id=?", (sport["sport_id"],)
            ).fetchone()["c"]
            r = conn.execute(
                "SELECT COUNT(*) c FROM game_results jr JOIN games g USING(game_id) "
                "WHERE g.sport_id=?", (sport["sport_id"],)
            ).fetchone()["c"]
            t = conn.execute(
                "SELECT COUNT(*) c FROM teams WHERE sport_id=?", (sport["sport_id"],)
            ).fetchone()["c"]
            span = conn.execute(
                "SELECT MIN(date_utc) a, MAX(date_utc) b FROM games WHERE sport_id=?",
                (sport["sport_id"],),
            ).fetchone()
            print(f"{sport['sport_id']:4} {sport['name']}")
            print(f"     games={g}  results={r}  teams={t}  span={span['a']} .. {span['b']}")
        print("\nrecent ingest runs:")
        for run in conn.execute(
            "SELECT job, sport_id, started_at, status, rows, error FROM ingest_runs "
            "ORDER BY run_id DESC LIMIT 5"
        ):
            err = f"  error={run['error'][:60]}" if run["error"] else ""
            print(f"  {run['started_at']}  {run['job']:16} {run['status']:8} rows={run['rows']}{err}")


def cmd_fit(args) -> None:
    matches = load_matches(args.sport)
    if matches.empty:
        raise SystemExit("no matches - run ingest first")
    fit = dixon_coles.fit(matches, half_life_days=args.half_life)

    strength = fit.attack - fit.defence
    # Long-relegated sides carry almost no decayed weight; their ratings are the
    # prior, not evidence, so they are excluded rather than shown as real.
    live = [i for i in range(len(fit.teams)) if fit.eff_weight[i] >= args.min_weight]
    order = sorted(live, key=lambda i: -strength[i])
    print(f"\nfitted {fit.n_matches} matches to {fit.as_of}  "
          f"(logL={fit.log_likelihood:.1f}, home_adv={fit.home_adv:.3f}, rho={fit.rho:.3f})")
    print(f"{len(live)} of {len(fit.teams)} teams above {args.min_weight:g} decayed matches\n")
    print(f"{'team':<18}{'attack':>9}{'defence':>9}{'net':>9}{'wgt':>8}")
    for i in order[: args.top]:
        print(f"{fit.teams[i]:<18}{fit.attack[i]:>9.3f}{fit.defence[i]:>9.3f}"
              f"{strength[i]:>9.3f}{fit.eff_weight[i]:>8.1f}")

    with connect() as conn:
        for i, team in enumerate(fit.teams):
            row = conn.execute(
                "SELECT team_id FROM teams WHERE sport_id=? AND name=?", (args.sport, team)
            ).fetchone()
            if row is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO team_ratings (sport_id, team_id, as_of, model_version, rating_json) "
                "VALUES (?,?,?,?,?)",
                (args.sport, row["team_id"], fit.as_of, dixon_coles.MODEL_VERSION,
                 json.dumps({"attack": float(fit.attack[i]), "defence": float(fit.defence[i])})),
            )
        conn.execute(
            "INSERT OR REPLACE INTO team_ratings (sport_id, team_id, as_of, model_version, rating_json) "
            "VALUES (?,?,?,?,?)",
            (args.sport, 0, fit.as_of, dixon_coles.MODEL_VERSION + ":global",
             json.dumps({"home_adv": fit.home_adv, "rho": fit.rho})),
        )
    print(f"\nratings written for {len(fit.teams)} teams")


def cmd_forecast(args) -> None:
    matches = load_matches(args.sport)
    if matches.empty:
        raise SystemExit("no matches - run ingest first")
    fit = dixon_coles.fit(matches, half_life_days=args.half_life)

    if args.match:
        home, away = [s.strip() for s in args.match.split("vs")]
        pairs = [(home, away)]
    else:
        # Most recent completed season's final matchday, as a stand-in fixture list
        # until the fixtures feed is wired in Phase 3.
        last = matches.sort_values("date").tail(args.n)
        pairs = list(zip(last["home"], last["away"]))

    print(f"\nmodel {dixon_coles.MODEL_VERSION}  fitted to {fit.as_of}  "
          f"({fit.n_matches} matches, home_adv={fit.home_adv:.3f}, rho={fit.rho:.3f})\n")
    print(f"{'fixture':<34}{'home':>8}{'draw':>8}{'away':>8}{'xG':>14}{'likeliest':>11}")
    print("-" * 83)
    for home, away in pairs:
        try:
            h, d, a = fit.outcome_probs(home, away)
            lam, mu = fit.rates(home, away)
            m = fit.score_matrix(home, away)
            i, j = np.unravel_index(np.argmax(m), m.shape)
        except KeyError as exc:
            print(f"{home} vs {away}: {exc}")
            continue
        print(f"{home + ' vs ' + away:<34}{h:>7.1%}{d:>8.1%}{a:>8.1%}"
              f"{lam:>7.2f}-{mu:<6.2f}{f'{i}-{j}':>11}")


def cmd_export_web(args) -> None:
    from pathlib import Path
    out = export_web(Path(args.out), sport_id=args.sport, season=args.season)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")


def cmd_backtest(args) -> None:
    df = bt.walk_forward(args.sport, start=args.start, refit_days=args.refit_days,
                         half_life=args.half_life)
    if df.empty:
        raise SystemExit("no predictions produced")
    res = bt.evaluate(df)
    cal = bt.calibration(df)
    ece = bt.expected_calibration_error(cal)

    span = f"{df['date'].min().date()}..{df['date'].max().date()}"
    print(f"\nwalk-forward  {span}  refit every {args.refit_days}d  "
          f"{len(df)} predictions\n")
    print(f"{'':<24}{'n':>7}{'Brier':>9}{'log loss':>11}{'accuracy':>11}")
    print("-" * 62)
    order = ["model", "market", "model_on_market_subset", "base_rate",
             "stronger_team", "home_always"]
    for k in order:
        if k not in res:
            continue
        r = res[k]
        b = f"{r['brier']:.4f}" if r.get("brier") is not None else "-"
        l = f"{r['log_loss']:.4f}" if r.get("log_loss") is not None else "-"
        print(f"{k:<24}{r['n']:>7}{b:>9}{l:>11}{r['accuracy']:>10.1%}")

    print(f"\ncalibration (all three outcomes pooled, ECE={ece:.4f})")
    print(f"{'bin':<12}{'n':>7}{'predicted':>11}{'observed':>10}{'gap':>9}")
    print("-" * 49)
    for r in cal.itertuples(index=False):
        print(f"{r.bin:<12}{r.n:>7}{r.predicted:>11.3f}{r.observed:>10.3f}{r.gap:>+9.3f}")

    if args.save:
        bt.persist(args.sport, res, cal, span, bt.per_season(df))
        print("\nwritten to model_evaluations")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"predictions written to {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(prog="hypz", description="HyPz sports simulator")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)

    s = sub.add_parser("ingest"); s.set_defaults(func=cmd_ingest)
    s.add_argument("--sport", default="pl", choices=ADAPTERS)
    s.add_argument("--seasons", help="comma-separated, e.g. 2024/25,2025/26")

    sub.add_parser("status").set_defaults(func=cmd_status)

    s = sub.add_parser("fit"); s.set_defaults(func=cmd_fit)
    s.add_argument("--sport", default="pl", choices=ADAPTERS)
    s.add_argument("--half-life", type=float, default=270.0)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--min-weight", type=float, default=5.0,
                   help="hide teams with less than this many decayed matches")

    s = sub.add_parser("forecast"); s.set_defaults(func=cmd_forecast)
    s.add_argument("--sport", default="pl", choices=ADAPTERS)
    s.add_argument("--half-life", type=float, default=270.0)
    s.add_argument("--match", help='e.g. "Arsenal vs Chelsea"')
    s.add_argument("-n", type=int, default=10)

    s = sub.add_parser("export-web"); s.set_defaults(func=cmd_export_web)
    s.add_argument("--sport", default="pl", choices=ADAPTERS)
    s.add_argument("--season", default="2026/27")
    s.add_argument("--out", default="/data/web/fixture-model.html")

    s = sub.add_parser("backtest"); s.set_defaults(func=cmd_backtest)
    s.add_argument("--sport", default="pl", choices=ADAPTERS)
    s.add_argument("--start", default="2012-08-01")
    s.add_argument("--refit-days", type=int, default=7)
    s.add_argument("--half-life", type=float, default=270.0)
    s.add_argument("--save", action="store_true")
    s.add_argument("--out", help="write per-match predictions to CSV")

    args = p.parse_args()
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
