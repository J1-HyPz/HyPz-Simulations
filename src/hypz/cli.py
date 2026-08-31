"""Command line entry point for the Phase 1 vertical slice."""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from .adapters.football_data import FootballDataAdapter
from .db import connect, init_db, now_iso
from . import health as health_mod
from . import lineups as lineups_mod
from . import matchcard
from .adapters import api_football as af
from .ingest import ingest_fixtures, ingest_results, known_teams, load_matches
from .predict import forecast_scheduled, scored_forecasts
from . import backtest as bt
from .export_web import export as export_web
from .models import dixon_coles

from . import leagues as leagues_mod

def adapter_for(sport_id: str) -> FootballDataAdapter:
    return FootballDataAdapter(sport_id)


ADAPTERS = leagues_mod.all_ids()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_init(args) -> None:
    init_db()
    print("schema initialised")


def cmd_leagues(args) -> None:
    from .db import connect as _c
    with _c() as conn:
        counts = {r["sport_id"]: r["n"] for r in conn.execute(
            "SELECT sport_id, COUNT(*) n FROM games GROUP BY sport_id")}
    print(f"\n{'id':<14}{'league':<18}{'country':<14}{'div':<6}{'teams':>6}{'games':>9}")
    print("-" * 68)
    for lid, lg in leagues_mod.LEAGUES.items():
        print(f"{lid:<14}{lg.name:<18}{lg.country:<14}{lg.code:<6}"
              f"{lg.teams:>6}{counts.get(lid, 0):>9}")


def _targets(args) -> list[str]:
    return leagues_mod.all_ids() if getattr(args, "all_leagues", False) else [args.sport]


def cmd_ingest(args) -> None:
    total, failed = 0, []
    seasons = args.seasons.split(",") if args.seasons else None
    for sid in _targets(args):
        try:
            n = ingest_results(adapter_for(sid), seasons, force=args.force)
        except Exception as exc:
            # One league's source being down must not cost the others their run.
            failed.append((sid, exc))
            print(f"  {sid:<14} FAILED  {str(exc)[:60]}")
            continue
        print(f"  {sid:<14} {n:>6} records")
        total += n
    print(f"ingested {total} records"
          + (f", {len(failed)} league(s) failed" if failed else ""))


def cmd_fixtures(args) -> None:
    for sid in _targets(args):
        try:
            n = ingest_fixtures(adapter_for(sid))
            print(f"  {sid:<14} {n:>4} fixtures")
        except Exception as exc:
            print(f"  {sid:<14} FAILED  {str(exc)[:60]}")


def cmd_predict(args) -> None:
    n = forecast_scheduled(args.sport)
    print(f"wrote {n} forecasts")
    if args.show:
        with connect() as conn:
            rows = conn.execute(
                "SELECT g.date_utc d, th.name h, ta.name a, f.home_win_prob hp,"
                " f.draw_prob dp, f.away_win_prob ap FROM forecasts f "
                "JOIN games g USING(game_id) "
                "JOIN teams th ON th.team_id=g.home_team_id "
                "JOIN teams ta ON ta.team_id=g.away_team_id "
                "WHERE g.status='scheduled' ORDER BY g.date_utc, th.name").fetchall()
        if rows:
            print(f"\n{'date':<12}{'fixture':<34}{'home':>8}{'draw':>8}{'away':>8}")
            print("-" * 70)
            for r in rows:
                print(f"{r['d']:<12}{r['h'] + ' vs ' + r['a']:<34}"
                      f"{r['hp']:>7.1%}{r['dp']:>8.1%}{r['ap']:>8.1%}")


def cmd_health(args) -> None:
    rows = health_mod.summary()
    state = health_mod.overall(rows)
    print(f"\npipeline: {state.upper()}\n")
    if not rows:
        print("  no runs recorded yet")
        return
    print(f"{'job':<18}{'state':<10}{'last run':<22}{'rows':>7}{'age(h)':>9}{'fails':>7}")
    print("-" * 73)
    for r in rows:
        age = f"{r['age_hours']:.1f}" if r["age_hours"] is not None else "-"
        print(f"{r['job']:<18}{r['state']:<10}{r['last_at'][:19]:<22}"
              f"{r['last_rows']:>7}{age:>9}{r['total_failures']:>7}")
        if r["last_error"]:
            print(f"  └ {r['last_error'][:88]}")
    raise SystemExit(0 if state == "ok" else 1)


def cmd_track(args) -> None:
    """Live track record: forecasts made before kickoff, scored after."""
    import numpy as np
    rows = scored_forecasts(args.sport)
    if not rows:
        print("no forecasts have been scored yet - none of the forecast fixtures "
              "have been played since they were predicted")
        return
    print(f"\n{len(rows)} scored forecast(s)\n")
    print(f"{'date':<12}{'fixture':<32}{'forecast H/D/A':<24}{'result':>8}{'brier':>8}")
    print("-" * 84)
    tot = 0.0
    for r in rows:
        act = ("H" if r["home_score"] > r["away_score"]
               else "D" if r["home_score"] == r["away_score"] else "A")
        p = np.array([r["home_win_prob"], r["draw_prob"], r["away_win_prob"]])
        y = np.array([act == o for o in "HDA"], dtype=float)
        b = float(((p - y) ** 2).sum()); tot += b
        probs = f"{p[0]:.0%}/{p[1]:.0%}/{p[2]:.0%}"
        print(f"{r['date_utc']:<12}{r['h'] + ' vs ' + r['a']:<32}{probs:<24}"
              f"{str(r['home_score']) + '-' + str(r['away_score']):>8}{b:>8.3f}")
    print(f"\nmean Brier: {tot/len(rows):.4f}   (backtest reference: 0.5777)")


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
        # Surface upstream gaps rather than letting them sit silently in the data.
        # football-data.co.uk ships 2003/04 and 2004/05 with 335 of 380 matches.
        expected = conn.execute(
            "SELECT config_json FROM sports WHERE sport_id='pl'").fetchone()
        import json as _json
        exp = _json.loads(expected["config_json"]).get("matches_per_season") if expected else None
        if exp:
            short = conn.execute(
                "SELECT season, COUNT(*) c FROM games WHERE status='final' "
                "GROUP BY season HAVING c < ? ORDER BY season", (exp,)).fetchall()
            cur = conn.execute(
                "SELECT MAX(season) m FROM games").fetchone()["m"]
            short = [r for r in short if r["season"] != cur]
            if short:
                print(f"\ndata gaps (source is short of {exp} matches):")
                for r in short:
                    print(f"  {r['season']}  {r['c']}/{exp}  "
                          f"({exp - r['c']} missing upstream)")

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
    fit = dixon_coles.fit(matches, half_life_days=args.half_life,
                          teams=known_teams(args.sport))

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

    from . import ratings as ratings_mod
    ratings_mod.persist(fit, args.sport)
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


def cmd_schedule(args) -> None:
    sched = matchcard.schedule(args.sport, days=args.days, today=args.start)
    print(f"\n{sched['count']} fixture(s), {sched['from']} to {sched['to']}")
    if not sched["count"]:
        print("  nothing scheduled in that window")
        return
    for day in sched["days"]:
        print(f"\n{date_heading(day['date'])}")
        for g in day["games"]:
            ko = g["kickoff"] or "--:--"
            line = f"  {ko}  {g['home']} v {g['away']}"
            if g["status"] == "final" and g["result"]:
                r = g["result"]["score"]
                line += f"   {r['home']}-{r['away']}  FT"
            elif g["forecast"]:
                f = g["forecast"]
                line += f"   {f['home']:.0%}/{f['draw']:.0%}/{f['away']:.0%}"
            print(line)
            fh = g["pre_match"]["form"]["home"]; fa = g["pre_match"]["form"]["away"]
            h2h = g["pre_match"]["head_to_head"]
            print(f"        form {fh['form'] or '-':<5} v {fa['form'] or '-':<5}"
                  f"   h2h {h2h['home_wins']}-{h2h['draws']}-{h2h['away_wins']}"
                  f"   rest {_rest(g['pre_match']['rest_days']['home'])}/"
                  f"{_rest(g['pre_match']['rest_days']['away'])}"
                  + (f"   ref {g['referee']}" if g["referee"] else ""))
            if g["status"] == "final" and g["result"]:
                print(f"        {g['result']['summary']}")


def _rest(v):
    return f"{v}d" if v is not None else "-"


def date_heading(d: str) -> str:
    from datetime import date as _d
    return _d.fromisoformat(d).strftime("%a %d %b")


def cmd_match(args) -> None:
    import json as _json
    c = matchcard.card(args.game_id, args.sport)
    if c is None:
        raise SystemExit(f"unknown game: {args.game_id}")
    print(_json.dumps(c, indent=2))


def cmd_lineups(args) -> None:
    from .adapters import highlightly as hl
    client = lineups_mod.get_provider(args.provider)
    if not client.configured:
        raise SystemExit(
            "No lineup provider is configured.\n\n"
            "  Recommended - Highlightly. Its free tier covers the current season.\n"
            "    1. Free key at https://highlightly.net/football-api/\n"
            "    2. Add to .env next to docker-compose.yml:\n"
            f"         {hl.KEY_ENV}=your-key-here\n"
            "    3. docker compose up -d\n\n"
            "  Alternative - API-Football. Note its free tier only reaches\n"
            "  seasons 2022-2024, so it cannot serve current fixtures.\n"
            f"         {af.KEY_ENV}=your-key-here")
    print(f"provider: {client.name}")
    if args.quota:
        q = client.quota()
        limit = q["limit"] if q["limit"] is not None else "unknown"
        print(f"plan {q['plan']}  used {q['used']} of {limit}")
        return
    mapped = lineups_mod.sync_fixture_ids(client) if args.sync else 0
    n = lineups_mod.fetch_lineups(client, lookahead_hours=args.hours)
    print(f"mapped {mapped} fixture id(s), stored {n} lineup(s), "
          f"{client.calls} api call(s)")
    with connect() as conn:
        un = conn.execute("SELECT raw_name, context FROM unmatched_names").fetchall()
    if un:
        print("\nunmatched names needing review:")
        for r in un:
            print(f"  {r['raw_name']}  ({r['context']})")


def main() -> None:
    p = argparse.ArgumentParser(prog="hypz", description="HyPz sports simulator")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)
    sub.add_parser("leagues").set_defaults(func=cmd_leagues)

    s = sub.add_parser("ingest"); s.set_defaults(func=cmd_ingest)
    s.add_argument("--sport", default=leagues_mod.DEFAULT, choices=ADAPTERS)
    s.add_argument("--seasons", help="comma-separated, e.g. 2024/25,2025/26")
    s.add_argument("--force", action="store_true", help="ignore the watermark, refetch all")
    s.add_argument("--all-leagues", action="store_true", help="every configured league")

    s = sub.add_parser("fixtures"); s.set_defaults(func=cmd_fixtures)
    s.add_argument("--sport", default=leagues_mod.DEFAULT, choices=ADAPTERS)
    s.add_argument("--all-leagues", action="store_true", help="every configured league")

    s = sub.add_parser("predict"); s.set_defaults(func=cmd_predict)
    s.add_argument("--sport", default=leagues_mod.DEFAULT, choices=ADAPTERS)
    s.add_argument("--show", action="store_true")

    sub.add_parser("health").set_defaults(func=cmd_health)

    s = sub.add_parser("track"); s.set_defaults(func=cmd_track)
    s.add_argument("--sport", default=leagues_mod.DEFAULT, choices=ADAPTERS)

    sub.add_parser("status").set_defaults(func=cmd_status)

    s = sub.add_parser("fit"); s.set_defaults(func=cmd_fit)
    s.add_argument("--sport", default=leagues_mod.DEFAULT, choices=ADAPTERS)
    s.add_argument("--half-life", type=float, default=270.0)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--min-weight", type=float, default=5.0,
                   help="hide teams with less than this many decayed matches")

    s = sub.add_parser("forecast"); s.set_defaults(func=cmd_forecast)
    s.add_argument("--sport", default=leagues_mod.DEFAULT, choices=ADAPTERS)
    s.add_argument("--half-life", type=float, default=270.0)
    s.add_argument("--match", help='e.g. "Arsenal vs Chelsea"')
    s.add_argument("-n", type=int, default=10)

    s = sub.add_parser("export-web"); s.set_defaults(func=cmd_export_web)
    s.add_argument("--sport", default=leagues_mod.DEFAULT, choices=ADAPTERS)
    s.add_argument("--season", default="2026/27")
    s.add_argument("--out", default="/data/web/fixture-model.html")

    s = sub.add_parser("backtest"); s.set_defaults(func=cmd_backtest)
    s.add_argument("--sport", default=leagues_mod.DEFAULT, choices=ADAPTERS)
    s.add_argument("--start", default="2012-08-01")
    s.add_argument("--refit-days", type=int, default=7)
    s.add_argument("--half-life", type=float, default=270.0)
    s.add_argument("--save", action="store_true")
    s.add_argument("--out", help="write per-match predictions to CSV")

    s = sub.add_parser("schedule"); s.set_defaults(func=cmd_schedule)
    s.add_argument("--sport", default=leagues_mod.DEFAULT, choices=ADAPTERS)
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--start", help="ISO date; defaults to today")

    s = sub.add_parser("match"); s.set_defaults(func=cmd_match)
    s.add_argument("game_id")
    s.add_argument("--sport", default=leagues_mod.DEFAULT, choices=ADAPTERS)

    s = sub.add_parser("lineups"); s.set_defaults(func=cmd_lineups)
    s.add_argument("--sync", action="store_true", help="refresh fixture id mapping first")
    s.add_argument("--hours", type=int, default=6, help="lookahead window")
    s.add_argument("--quota", action="store_true", help="report remaining daily allowance")
    s.add_argument("--provider", choices=["highlightly", "api_football"],
                   help="override provider selection")

    args = p.parse_args()
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
