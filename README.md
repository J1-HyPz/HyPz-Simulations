# HyPz-Simulations

Probabilistic forecasts for football and NFL fixtures. The output is always a
distribution — *"Arsenal win 45.6% of the time, likeliest score 1–1"* — never a
single predicted scoreline.

**Status:** Phases 1–2 complete. Premier League forecasts are produced end to end
and validated by walk-forward backtest against the market closing line.

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
| 3 | Scheduling, watermarks, health dashboard | next |
| 4 | NFL (nflverse parquet, drive model) | |
| 5 | Read API and web UI | partial — static page ships now |

The model is calibrated and beats its baselines. What it is not is a market edge,
and nothing in the roadmap is aimed at becoming one.
