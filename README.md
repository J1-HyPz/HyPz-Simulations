# HyPz-Simulations

Probabilistic forecasts for football and NFL fixtures. The output is always a
distribution — *"Arsenal win 45.6% of the time, likeliest score 1–1"* — never a
single predicted scoreline.

**Status:** Phase 1 complete. Premier League ingestion and a fitted Dixon–Coles
model produce match forecasts end to end. Not yet evaluated — see Phase 2.

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

`status` reports row counts and recent ingest runs.

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
form: a bivariate Poisson grid over 0–10 goals, corrected and normalised. The
design doc's 20,000-iteration simulator is reserved for NFL, whose drive model has
no analytic equivalent.

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

## Roadmap

| Phase | | Status |
|---|---|---|
| 1 | PL vertical slice | ✅ done |
| 2 | Walk-forward backtest, calibration, baselines | next |
| 3 | Scheduling, watermarks, health dashboard | |
| 4 | NFL (nflverse parquet, drive model) | |
| 5 | Read API and web UI | partial — static page ships now |

Phase 2 is the one that matters. Until a walk-forward backtest says otherwise,
these forecasts are unvalidated — a model can look entirely reasonable and still
be worse than "the home team wins".
