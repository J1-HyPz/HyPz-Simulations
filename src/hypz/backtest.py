"""Walk-forward evaluation (design doc section 9).

Train on everything before date T, predict the fixtures on T, advance. The whole
point is that no information from on or after the prediction date may reach the
model - this is the single easiest way to build something that looks brilliant and
is worthless, so the cutoff is enforced in two independent places: `dixon_coles.fit`
filters `date < as_of`, and `walk_forward` asserts it again per prediction.
"""
from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd

from .db import connect, now_iso
from .ingest import load_matches
from .models import dixon_coles

log = logging.getLogger(__name__)

OUTCOMES = ("H", "D", "A")
EPS = 1e-15


def actual_outcome(hs: int, as_: int) -> str:
    return "H" if hs > as_ else ("D" if hs == as_ else "A")


def devig(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Decimal odds to probabilities, normalised to remove the bookmaker margin."""
    inv = np.array([1.0 / h, 1.0 / d, 1.0 / a])
    return tuple(inv / inv.sum())


def walk_forward(sport_id: str = "pl", start: str = "2012-08-01",
                 refit_days: int = 7, half_life: float = 270.0) -> pd.DataFrame:
    matches = load_matches(sport_id)
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    # Fixed team list keeps parameter indices stable so warm starts remain valid.
    teams = sorted(set(matches["home"]) | set(matches["away"]))

    # Outcomes once, up front: the base-rate baseline then comes from cumulative
    # counts rather than re-scanning history at every match date.
    matches["outcome"] = np.where(
        matches["home_score"] > matches["away_score"], "H",
        np.where(matches["home_score"] == matches["away_score"], "D", "A"))
    cum = {o: (matches["outcome"] == o).cumsum().to_numpy() for o in OUTCOMES}
    dates_sorted = matches["date"].to_numpy()

    start_ts = pd.Timestamp(start)
    eval_dates = sorted(matches.loc[matches["date"] >= start_ts, "date"].unique())
    log.info("walk-forward over %d match dates from %s", len(eval_dates), start)

    rows: list[dict] = []
    fit = None
    fit_as_of: pd.Timestamp | None = None
    warm = None
    n_refits = 0

    for d in eval_dates:
        d = pd.Timestamp(d)
        if fit is None or (d - fit_as_of).days >= refit_days:
            fit = dixon_coles.fit(matches, as_of=d, half_life_days=half_life,
                                  x0=warm, teams=teams)
            warm = fit.raw_params
            fit_as_of = d
            n_refits += 1
            if n_refits % 50 == 0:
                log.info("  %d refits, at %s", n_refits, d.date())

        # Leakage guard, independent of the filter inside fit().
        assert pd.Timestamp(fit.as_of) <= d, (
            f"training cutoff {fit.as_of} is not before prediction date {d.date()}")

        # Base rate from history only, at the same cutoff.
        k = int(np.searchsorted(dates_sorted, d.to_datetime64(), side="left"))
        if k < 200:
            continue
        base = tuple(float(cum[o][k - 1] / k) for o in OUTCOMES)

        tix = {t: i for i, t in enumerate(fit.teams)}
        for r in matches[matches["date"] == d].itertuples(index=False):
            try:
                ph, pd_, paw = fit.outcome_probs(r.home, r.away)
            except KeyError:
                continue
            extra = json.loads(r.extra_json) if r.extra_json else {}
            mkt = (None, None, None)
            if all(k in extra for k in ("close_h", "close_d", "close_a")):
                try:
                    mkt = devig(extra["close_h"], extra["close_d"], extra["close_a"])
                except (ZeroDivisionError, ValueError):
                    mkt = (None, None, None)
            ih, ia = tix[r.home], tix[r.away]
            net_h = fit.attack[ih] - fit.defence[ih]
            net_a = fit.attack[ia] - fit.defence[ia]
            rows.append({
                "date": d, "season": r.season, "home": r.home, "away": r.away,
                "actual": actual_outcome(r.home_score, r.away_score),
                "train_through": fit.as_of, "train_n": fit.n_matches,
                "p_h": ph, "p_d": pd_, "p_a": paw,
                "b_h": base[0], "b_d": base[1], "b_a": base[2],
                "m_h": mkt[0], "m_d": mkt[1], "m_a": mkt[2],
                "stronger": "H" if net_h + fit.home_adv >= net_a else "A",
            })

    df = pd.DataFrame(rows)
    log.info("%d predictions from %d refits", len(df), n_refits)

    # Final audit: every training cutoff must precede its prediction date.
    bad = df[pd.to_datetime(df["train_through"]) > df["date"]]
    if len(bad):
        raise RuntimeError(f"LEAKAGE: {len(bad)} predictions trained on same-or-later data")
    return df


def _scores(p: np.ndarray, y: np.ndarray) -> dict:
    """p: (n,3) probabilities in H,D,A order. y: (n,3) one-hot."""
    brier = float(np.mean(np.sum((p - y) ** 2, axis=1)))
    ll = float(-np.mean(np.log(np.clip(p[y.astype(bool)], EPS, 1.0))))
    acc = float(np.mean(np.argmax(p, axis=1) == np.argmax(y, axis=1)))
    return {"brier": brier, "log_loss": ll, "accuracy": acc, "n": int(len(p))}


def _onehot(actual: pd.Series) -> np.ndarray:
    return np.stack([(actual == o).to_numpy(dtype=float) for o in OUTCOMES], axis=1)


def evaluate(df: pd.DataFrame) -> dict:
    y = _onehot(df["actual"])
    out = {"model": _scores(df[["p_h", "p_d", "p_a"]].to_numpy(dtype=float), y),
           "base_rate": _scores(df[["b_h", "b_d", "b_a"]].to_numpy(dtype=float), y)}

    # Accuracy-only baselines: degenerate probabilities make log loss infinite.
    out["home_always"] = {"accuracy": float((df["actual"] == "H").mean()), "n": len(df),
                          "brier": None, "log_loss": None}
    out["stronger_team"] = {"accuracy": float((df["actual"] == df["stronger"]).mean()),
                            "n": len(df), "brier": None, "log_loss": None}

    sub = df.dropna(subset=["m_h", "m_d", "m_a"])
    if len(sub):
        ys = _onehot(sub["actual"])
        out["market"] = _scores(sub[["m_h", "m_d", "m_a"]].to_numpy(dtype=float), ys)
        # Model restricted to the same fixtures, so the comparison is like for like.
        out["model_on_market_subset"] = _scores(
            sub[["p_h", "p_d", "p_a"]].to_numpy(dtype=float), ys)
    return out


def calibration(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Pool all three outcomes: of everything forecast at p, how often did it happen?"""
    p = df[["p_h", "p_d", "p_a"]].to_numpy(dtype=float).ravel()
    y = _onehot(df["actual"]).ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                     "n": int(m.sum()),
                     "predicted": float(p[m].mean()),
                     "observed": float(y[m].mean())})
    cal = pd.DataFrame(rows)
    cal["gap"] = cal["observed"] - cal["predicted"]
    return cal


def expected_calibration_error(cal: pd.DataFrame) -> float:
    return float((cal["n"] / cal["n"].sum() * cal["gap"].abs()).sum())


def per_season(df: pd.DataFrame) -> pd.DataFrame:
    """Model vs market Brier by season, on the fixtures where both exist."""
    sub = df.dropna(subset=["m_h", "m_d", "m_a"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["season", "n", "model", "market", "gap"])
    y = _onehot(sub["actual"])
    sub["_model"] = ((sub[["p_h", "p_d", "p_a"]].to_numpy(dtype=float) - y) ** 2).sum(1)
    sub["_market"] = ((sub[["m_h", "m_d", "m_a"]].to_numpy(dtype=float) - y) ** 2).sum(1)
    g = (sub.groupby("season")
            .agg(n=("_model", "size"), model=("_model", "mean"), market=("_market", "mean"))
            .reset_index())
    g["gap"] = g["model"] - g["market"]
    return g


def persist(sport_id: str, results: dict, cal: pd.DataFrame, window: str,
            seasons: pd.DataFrame | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS model_evaluations ("
            "eval_id INTEGER PRIMARY KEY AUTOINCREMENT, sport_id TEXT, model_version TEXT,"
            "window TEXT, run_at TEXT, brier REAL, log_loss REAL, accuracy REAL,"
            "n INTEGER, calibration_json TEXT, baselines_json TEXT, seasons_json TEXT)")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(model_evaluations)")}
        if "seasons_json" not in cols:
            conn.execute("ALTER TABLE model_evaluations ADD COLUMN seasons_json TEXT")
        m = results["model"]
        conn.execute(
            "INSERT INTO model_evaluations (sport_id, model_version, window, run_at,"
            " brier, log_loss, accuracy, n, calibration_json, baselines_json, seasons_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (sport_id, dixon_coles.MODEL_VERSION, window, now_iso(),
             m["brier"], m["log_loss"], m["accuracy"], m["n"],
             cal.to_json(orient="records"),
             json.dumps({k: v for k, v in results.items() if k != "model"}),
             seasons.to_json(orient="records") if seasons is not None else None))
