# HyPz-Simulations

Probabilistic forecasts for football and NFL fixtures. The output is always a
distribution — *"Arsenal win 45.6% of the time, likeliest score 1–1"* — never a
single predicted scoreline.

**Status:** Eight leagues, Phases 1–3 complete. Premier League forecasts run unattended on a
schedule, are recorded before kickoff, and are validated by walk-forward backtest
against the market closing line.

## Leagues

| id | League | Country | Division | Matches |
|---|---|---|---|---|
| `pl` | Premier League | England | E0 | 12,634 |
| `laliga` | La Liga | Spain | SP1 | 12,612 |
| `seriea` | Serie A | Italy | I1 | 11,544 |
| `bundesliga` | Bundesliga | Germany | D1 | 9,970 |
| `ligue1` | Ligue 1 | France | F1 | 11,510 |
| `eredivisie` | Eredivisie | Netherlands | N1 | 9,760 |
| `primeira` | Primeira Liga | Portugal | P1 | 9,412 |
| `championship` | Championship | England | E1 | 17,824 |

95,266 matches, 420 teams, all back to 1993/94, each fitted separately. Adding one
is a row in `leagues.py` — the adapter contract keeps the core ignorant of any
particular competition and every table is already keyed by `sport_id`.

Fitted home advantage varies more than expected: Serie A 0.120 against La Liga
0.314. Fitting per league rather than pooling is what surfaces that.

### Why there is no Champions League

Two reasons, and the second is the real one.

football-data.co.uk covers domestic leagues only — the UCL division 404s. That part
is just a missing feed.

The harder problem is that **ratings fitted per league are not comparable across
leagues**. Each fit constrains its own attack parameters to sum to zero, so
"+0.4" in the Eredivisie and "+0.4" in La Liga are separate scales, not one.
Forecasting Arsenal against Real Madrid needs both on a common footing, which
means either a joint fit with league-strength offsets estimated from the matches
that actually cross leagues, or fitting on continental results directly — where
each team plays six to thirteen games a season. That is a modelling change, not a
data-plumbing one, and doing it badly would produce confident nonsense.

## Scope

Narrowed from the original five-sport design ([`sports-simulator-design.md`](sports-simulator-design.md))
to **Premier League and NFL**. The local-LLM injury extraction service was dropped:
it depended on an unsolved data-sourcing problem and a GPU, for a small expected gain.

## Quick start

```bash
docker compose up -d
docker exec hypz-sim python -m hypz.cli init
docker exec hypz-sim python -m hypz.cli ingest --sport pl
docker exec hypz-sim python -m hypz.cli fit --sport pl
docker exec hypz-sim python -m hypz.cli forecast --sport pl --match "Arsenal vs Man City"
```

`status` reports row counts, upstream data gaps and recent ingest runs;
`health` reports per-job pipeline state and exits non-zero when anything is stale
or failing, so it works as a check in a monitor.

## The app

```bash
docker compose up -d
```

Then open **http://<host>:8099** — the forecaster page, plus a read-only JSON API
at `/api/docs`.

| Endpoint | |
|---|---|
| `GET /` | the forecaster page |
| `GET /api/forecast?home=X&away=Y` | any matchup: probabilities, expected goals, likeliest scorelines |
| `GET /api/schedule?days=7` | fixtures from today onward, each with its full card |
| `GET /api/match/{game_id}` | one fixture: form, head to head, lineups, stats, summary |
| `GET /api/fixtures` | scheduled fixtures with their pre-kickoff forecast |
| `GET /api/teams` | fitted ratings |
| `GET /api/evaluation` | latest walk-forward result |
| `GET /api/health` | pipeline state; 503 when stale or failing |

Serving runs in its own container. That is design doc section 3's boundary made
real: ingestion writes, serving reads, they share only the database, and either
can be down without the other caring. The API never fits a model on request — it
reloads ratings written by the scheduled `model.ratings` job — so a slow source
can never make a page request hang.

## The page

Four sections behind a sticky nav — **Fixtures**, **Forecast**, **Ratings**,
**Accuracy** — which is the order someone actually reads it in: what is on, try it
yourself, who the model rates, and how well it has done. Model provenance sits in
the footer rather than the header; `rho = -0.084` is not a headline.

Ratings are drawn as diverging bars from a common zero rather than a table of
decimals, so the spread between sides is the thing you read. Recent results and
upcoming fixtures share one schedule instead of two panels repeating each other.

## The schedule

```bash
docker exec hypz-sim python -m hypz.cli schedule --days 7
```

Fixtures from today onward. Each carries, drawn entirely from data already held:

- **Form** — last five matches per side, bounded to 400 days so a promoted team
  shows "no recent matches in this division" rather than results from a previous spell
- **Venue record** — home form for the home side, away form for the away side,
  because a combined table hides the thing you want to know
- **Head to head** — last six meetings, counted from the upcoming home side's view
- **Rest days** — suppressed beyond 45 days, since a summer break or an absence
  from the division is not rest
- **Referee**, kickoff time, and the pre-kickoff model forecast
- Once played: **shots, shots on target, corners, fouls, cards, half-time score**
  and a generated factual summary

### Lineups

Optional. With no key the pipeline runs unchanged and the page says lineups are
unavailable and why.

```bash
cp .env.example .env      # paste a key into it
docker compose up -d
docker exec hypz-sim python -m hypz.cli lineups --sync
```

Two providers are supported behind one contract, and selection is automatic:

| Provider | Free tier | Verdict |
|---|---|---|
| **Highlightly** (default) | 100 req/day, all leagues, **current season** | recommended |
| API-Football | 100 req/day, **seasons 2022–2024 only** | cannot serve current fixtures |

API-Football gates seasons by plan; Highlightly gates only request volume, so its
free tier reaches the current season. `HYPZ_LINEUP_PROVIDER` forces one if needed.

They disagree at the wire level — API-Football takes a date range and returns a
flat XI, Highlightly takes one date at a time and nests the XI one row per
formation line — so both are normalised behind `adapters/lineup_provider.py` and
`hypz.lineups` never learns which is in use.

Roughly thirteen requests per matchweek on Highlightly: one to resolve the league,
one per match date, then one per fixture, never repeated once a lineup is stored.

Before kickoff the endpoint answers with a well-formed placeholder — empty arrays
and formation `"Unknown"` — rather than an error. That is treated as absent, not
stored, and retried later; caching it would satisfy the "already have it" check
forever and the real lineup would never arrive.

Team names differ between sources ("Manchester United" against "Man United"), so
they are resolved once by alias then fuzzy match and stored on the team row.
Unresolved names go to `unmatched_names` for review rather than being guessed at.
All 20 current Premier League names resolve with none queued.

### Still not there

**Live in-play state.** The results feed publishes only after full time. This is
reported in the API response under `unavailable` rather than being quietly missing.

## Running unattended

The container's main process is an APScheduler daemon:

| Job | Cadence |
|---|---|
| `ingest.results` | 04:00 daily |
| `ingest.fixtures` | 08:00 and 20:00 |
| `ingest.lineups` | hourly at :20 (only spends a call when a fixture is imminent) |
| `model.ratings` | 04:45 daily |
| `model.forecast` | 05:00 daily |
| `export.web` | 05:15 daily |

Every job is wrapped so an exception is logged and recorded in `ingest_runs`
rather than killing the scheduler — a dead scheduler is silent, a failed run is
visible. Results ingestion reads its watermark and fetches only seasons that are
not yet finished, which takes the run from ~7.5s to ~1.2s.

Forecasts for scheduled fixtures are written to `forecasts` before kickoff and
fingerprinted with an `inputs_hash`, so a re-run with unchanged ratings writes
nothing. `track` scores them once the matches are played — a prospective record,
which is stronger evidence than any backtest.

## Tests

```bash
docker exec hypz-sim python -m pytest /app/tests -q
```

77 tests, about two seconds. They cover the analytic gradient against numerical
differencing, the `as_of` cutoff that the whole backtest depends on, ingest
idempotency and watermark contiguity, the leakage audit predicate, and two
regressions for bugs found in Phase 3 (a stale watermark format, and a BOM that
`latin-1` decoding turned into mojibake), and static checks on the page template —
every id the script touches must exist, and the outcome colour classes must be
reachable from both places that use them.

## The forecaster page

```bash
docker exec hypz-sim python -m hypz.cli export-web
```

Builds a single self-contained HTML file — no backend, no build step, nothing to
install. The Dixon–Coles grid is closed form, so the page ships the fitted ratings
and recomputes every matchup in the browser: pick any two teams and the scoreline
distribution, outcome split and expected goals update live.

The score matrix uses composite encoding — hue for the outcome, depth for the
probability — so the draw diagonal falls out of the geometry rather than being
drawn on. Re-run the command after any refit and the page regenerates.

The JavaScript implementation is checked against the Python one: both produce
identical probabilities to the displayed precision across every fixture tested.

## How it works

**Data.** Premier League results come from football-data.co.uk — one plain CSV per
season back to 1993/94. No scraping, no API key, no rate limit. Every response is
written to the raw store before parsing, so a parser fix never needs a refetch.
Currently 12,624 matches across 34 seasons.

**Model.** Dixon–Coles (1997) bivariate Poisson. Each team carries an attack and a
defence rating; expected goals are `exp(attack_home + defence_away + home_adv)` and
`exp(attack_away + defence_home)`. Two corrections matter:

- **Low-score dependence.** Independent Poisson understates 0–0, 1–0, 0–1 and 1–1.
  The `rho` term corrects it; the fit lands around −0.08, consistent with the literature.
- **Time decay.** Matches are exponentially down-weighted with a 270-day half-life,
  so the fit tracks current form rather than 1990s results.

Ratings are shrunk toward the league mean by a Gaussian prior. Without it, teams
long relegated have effectively zero weight, and the optimiser pushes their ratings
to whatever bound it likes — a nonsense that shows up immediately as a Championship
side topping the table.

**No Monte Carlo.** For a match-goals sport the scoreline distribution is closed
form: a bivariate Poisson grid over 0–10 goals, corrected and normalised. Measured
across every current fixture pairing, the worst-case probability mass falling outside
that grid is 3.7e-04. The design doc's 20,000-iteration simulator is reserved for NFL,
whose drive model has no analytic equivalent.

## Layout

```
src/hypz/
  config.py               paths and tunables
  db.py                   SQLite schema, ingest instrumentation, watermarks
  ingest.py               idempotent upsert pipeline
  adapters/base.py        the SportAdapter contract
  adapters/football_data.py   Premier League
  models/dixon_coles.py   the model
  cli.py                  init / ingest / status / fit / forecast
```

Adding a sport means writing an adapter, not touching the engine.

## Validation

```bash
docker exec hypz-sim python -m hypz.cli backtest --start 2012-08-01 --refit-days 7 --save
```

Walk-forward across 5,330 matches, 2012/13 to date, refitting weekly. Every forecast
uses only matches played strictly *before* its own date — enforced in two independent
places, because leakage is the failure mode that makes a worthless model look brilliant.

| Forecaster | n | Brier | Log loss | Accuracy |
|---|---|---|---|---|
| Market closing line | 5,150 | **0.5633** | **0.9523** | 55.2% |
| This model | 5,330 | 0.5777 | 0.9730 | 53.7% |
| This model, market fixtures only | 5,150 | 0.5755 | 0.9700 | 54.1% |
| Base rate | 5,330 | 0.6467 | 1.0694 | 44.6% |
| Home team wins | 5,330 | — | — | 44.6% |

Both shipping bars from the design doc are cleared: the model beats the base rate on
every proper scoring rule, and beats "the home team wins" by nine accuracy points. It
does not beat the closing line, and was never expected to — the gap is stable at
0.01–0.02 Brier in 13 of 14 seasons. A model that suddenly beat the market would be
better evidence of a leak than of an edge.

Calibration is the number that matters for using the output: **ECE 0.0075**, with the
0.0–0.6 range accurate to within 0.016.

### Accuracy is close to useless here

The model names a draw as the single likeliest result **24 times in 5,330**, though
draws occur 23.8% of the time. A draw's probability tops out near 0.37 while a favoured
side's exceeds it, so the argmax is 99.5% identical to "the stronger team wins". Judge
this model on Brier, log loss and calibration.

## Roadmap

| Phase | | Status |
|---|---|---|
| 1 | PL vertical slice | ✅ done |
| 2 | Walk-forward backtest, calibration, baselines | ✅ done |
| 3 | Scheduling, watermarks, health, fixtures, tests | ✅ done |
| 4 | NFL (nflverse parquet, drive model) | next |
| 5 | Read API and web UI | ✅ done · Postgres still open |

The model is calibrated and beats its baselines. What it is not is a market edge,
and nothing in the roadmap is aimed at becoming one.
