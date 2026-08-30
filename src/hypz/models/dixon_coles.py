"""Dixon-Coles bivariate Poisson with exponential time decay.

Dixon & Coles (1997), "Modelling Association Football Scores and Inefficiencies
in the Football Betting Market". Two departures from naive independent Poisson:

  1. A low-score correction (tau) for 0-0, 1-0, 0-1 and 1-1, where independence
     demonstrably fails - draws are more common than independent Poisson implies.
  2. Exponential down-weighting of old matches, so the fit tracks current form.

Note what is absent: any Monte Carlo loop. For a match-goals sport the scoreline
distribution is available in closed form over a small grid, so section 7.1's
20,000-iteration simulator buys nothing here. It is reserved for NFL, whose drive
model has no analytic equivalent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from ..config import HALF_LIFE_DAYS, MAX_GOALS

log = logging.getLogger(__name__)

MODEL_VERSION = "dixon-coles-1.1"

# Precision of the Gaussian prior shrinking ratings toward the league mean.
# Roughly "this many decayed matches of evidence before a team moves freely".
REG_STRENGTH = 5.0

# tau can go non-positive for extreme rho; floor it so the log-likelihood stays finite.
_TAU_FLOOR = 1e-10


def _tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score dependence correction."""
    t = np.ones_like(lam, dtype=float)
    m = (x == 0) & (y == 0)
    t[m] = 1.0 - lam[m] * mu[m] * rho
    m = (x == 0) & (y == 1)
    t[m] = 1.0 + lam[m] * rho
    m = (x == 1) & (y == 0)
    t[m] = 1.0 + mu[m] * rho
    m = (x == 1) & (y == 1)
    t[m] = 1.0 - rho
    return np.maximum(t, _TAU_FLOOR)


@dataclass
class DixonColesFit:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    as_of: str
    n_matches: int
    log_likelihood: float
    eff_weight: np.ndarray  # time-decayed matches behind each team's rating

    def active_teams(self, min_weight: float = 5.0) -> list[str]:
        """Teams with enough recent evidence for their rating to mean anything."""
        return [t for t, w in zip(self.teams, self.eff_weight) if w >= min_weight]

    def _idx(self, team: str) -> int:
        try:
            return self.teams.index(team)
        except ValueError as exc:
            raise KeyError(f"team not in fitted ratings: {team!r}") from exc

    def rates(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals for (home, away)."""
        h, a = self._idx(home), self._idx(away)
        lam = float(np.exp(self.attack[h] + self.defence[a] + self.home_adv))
        mu = float(np.exp(self.attack[a] + self.defence[h]))
        return lam, mu

    def score_matrix(self, home: str, away: str, max_goals: int = MAX_GOALS) -> np.ndarray:
        """P(home=i, away=j) over an (n+1) x (n+1) grid, normalised."""
        lam, mu = self.rates(home, away)
        g = np.arange(max_goals + 1)
        # Poisson pmf via logs, then the tau correction on the 2x2 low-score block.
        log_h = g * np.log(lam) - lam - gammaln(g + 1)
        log_a = g * np.log(mu) - mu - gammaln(g + 1)
        m = np.exp(log_h[:, None] + log_a[None, :])
        m[0, 0] *= 1.0 - lam * mu * self.rho
        m[0, 1] *= 1.0 + lam * self.rho
        m[1, 0] *= 1.0 + mu * self.rho
        m[1, 1] *= 1.0 - self.rho
        m = np.maximum(m, 0.0)
        return m / m.sum()

    def outcome_probs(self, home: str, away: str) -> tuple[float, float, float]:
        """(home win, draw, away win)."""
        m = self.score_matrix(home, away)
        draw = float(np.trace(m))
        home_win = float(np.tril(m, -1).sum())  # rows = home goals, so below diagonal
        away_win = float(np.triu(m, 1).sum())
        return home_win, draw, away_win


def fit(matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
        half_life_days: float = HALF_LIFE_DAYS, reg: float = REG_STRENGTH) -> DixonColesFit:
    """Fit by weighted maximum likelihood.

    `matches` needs columns: date, home, away, home_score, away_score.
    Only matches strictly before `as_of` are used - the walk-forward backtest in
    Phase 2 depends on that being airtight, so the cutoff lives here rather than
    in each caller.
    """
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
    df = df[df["date"] < as_of]
    if df.empty:
        raise ValueError("no matches before as_of")

    teams = sorted(set(df["home"]) | set(df["away"]))
    n = len(teams)
    index = {t: i for i, t in enumerate(teams)}

    hi = df["home"].map(index).to_numpy()
    ai = df["away"].map(index).to_numpy()
    hg = df["home_score"].to_numpy(dtype=int)
    ag = df["away_score"].to_numpy(dtype=int)

    age_days = (as_of - df["date"]).dt.total_seconds().to_numpy() / 86400.0
    xi = np.log(2.0) / half_life_days
    w = np.exp(-xi * age_days)

    # Decayed match count behind each team, home and away appearances alike.
    eff_weight = np.bincount(hi, weights=w, minlength=n) + np.bincount(ai, weights=w, minlength=n)

    lg_hg = gammaln(hg + 1)
    lg_ag = gammaln(ag + 1)

    # Free parameters: attack[0..n-2], defence[0..n-1], home_adv, rho.
    # attack[n-1] is pinned to -sum(others) because adding a constant to every
    # attack and subtracting it from every defence leaves the rates unchanged.
    def unpack(p: np.ndarray):
        att_free = p[: n - 1]
        attack = np.concatenate([att_free, [-att_free.sum()]])
        defence = p[n - 1 : 2 * n - 1]
        return attack, defence, p[-2], p[-1]

    def neg_log_lik(p: np.ndarray) -> float:
        attack, defence, home_adv, rho = unpack(p)
        lam = np.exp(attack[hi] + defence[ai] + home_adv)
        mu = np.exp(attack[ai] + defence[hi])
        ll = (
            np.log(_tau(hg, ag, lam, mu, rho))
            + hg * np.log(lam) - lam - lg_hg
            + ag * np.log(mu) - mu - lg_ag
        )
        # Gaussian prior at the league mean (design doc section 8, step 3). Teams
        # with plenty of recent matches barely feel it; teams with none - long
        # relegated, so weight ~0 - are held at average instead of drifting to a
        # bound where the likelihood is flat and any value fits equally well.
        penalty = 0.5 * reg * (np.sum(attack ** 2) + np.sum(defence ** 2))
        return -float(np.sum(w * ll)) + penalty

    x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25], [-0.05]])
    bounds = [(-3, 3)] * (n - 1) + [(-3, 3)] * n + [(-1, 1), (-0.2, 0.2)]

    res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "maxfun": 200_000})
    if not res.success:
        log.warning("optimiser did not converge cleanly: %s", res.message)

    attack, defence, home_adv, rho = unpack(res.x)
    log.info(
        "fit %d matches, %d teams, home_adv=%.3f rho=%.3f", len(df), n, home_adv, rho
    )
    return DixonColesFit(
        teams=teams,
        attack=attack,
        defence=defence,
        home_adv=float(home_adv),
        rho=float(rho),
        as_of=as_of.date().isoformat(),
        n_matches=int(len(df)),
        log_likelihood=float(-res.fun),
        eff_weight=eff_weight,
    )
