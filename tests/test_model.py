"""Model correctness. The gradient test exists because a wrong gradient converges
quietly to the wrong answer - it produces a plausible model, not a crash."""
import numpy as np
import pandas as pd
import pytest
import scipy.optimize as so
from scipy.optimize import check_grad

from hypz.models import dixon_coles


def _capture_objective(matches):
    """Grab the real closure the fitter optimises, so the test covers shipped code."""
    cap = {}
    orig = so.minimize

    def spy(fun, x0, **kw):
        cap["fun"], cap["x0"] = fun, x0
        return orig(fun, x0, **kw)

    dixon_coles.minimize = spy
    try:
        dixon_coles.fit(matches)
    finally:
        dixon_coles.minimize = orig
    return cap


def test_analytic_gradient_matches_numerical(synthetic_matches):
    cap = _capture_objective(synthetic_matches)
    f = cap["fun"]
    obj, jac = (lambda p: f(p)[0]), (lambda p: f(p)[1])
    rng = np.random.default_rng(0)
    for _ in range(6):
        p = cap["x0"] + rng.normal(0, 0.3, size=len(cap["x0"]))
        p[-1] = float(np.clip(p[-1], -0.18, 0.18))
        err = check_grad(obj, jac, p, epsilon=1e-7)
        scale = max(float(np.linalg.norm(jac(p))), 1e-12)
        assert err / scale < 1e-5, f"relative gradient error {err/scale:.2e}"


def test_gradient_finite_where_tau_floor_binds(synthetic_matches):
    """Regression: the floor makes the objective flat, so the gradient must be
    finite there. It previously returned ~1e11."""
    cap = _capture_objective(synthetic_matches)
    f = cap["fun"]
    p = cap["x0"].copy()
    p[-1] = 0.18          # large rho drives tau non-positive on low scorelines
    p[-2] = 0.9           # and a large home advantage inflates lambda
    obj, grad = f(p)
    assert np.isfinite(obj)
    assert np.all(np.isfinite(grad))
    assert np.linalg.norm(grad) < 1e6, "gradient exploded where the tau floor binds"


def test_score_matrix_is_a_distribution(synthetic_matches):
    fit = dixon_coles.fit(synthetic_matches)
    for h, a in [("A", "D"), ("D", "A"), ("B", "C")]:
        m = fit.score_matrix(h, a)
        assert (m >= 0).all()
        assert m.sum() == pytest.approx(1.0, abs=1e-12)
        probs = fit.outcome_probs(h, a)
        assert sum(probs) == pytest.approx(1.0, abs=1e-12)


def test_stronger_team_is_favoured(synthetic_matches):
    fit = dixon_coles.fit(synthetic_matches)
    h_strong, _, a_strong = fit.outcome_probs("A", "D")
    h_weak, _, a_weak = fit.outcome_probs("D", "A")
    assert h_strong > a_strong, "strong home side should be favoured"
    assert a_weak > h_weak, "strong away side should be favoured despite home advantage"


def test_fit_excludes_matches_on_or_after_as_of(synthetic_matches):
    """The single most important invariant in the project."""
    cutoff = pd.Timestamp("2021-06-01")
    fit = dixon_coles.fit(synthetic_matches, as_of=cutoff)
    before = (synthetic_matches["date"] < cutoff).sum()
    assert fit.n_matches == before
    assert pd.Timestamp(fit.as_of) == cutoff


def test_shrinkage_holds_unseen_teams_at_the_mean(synthetic_matches):
    """A team with no matches must sit at the prior, not drift to a bound."""
    fit = dixon_coles.fit(synthetic_matches, teams=["A", "B", "C", "D", "GHOST"])
    i = fit.teams.index("GHOST")
    assert fit.eff_weight[i] == pytest.approx(0.0)
    assert abs(fit.attack[i]) < 0.05
    assert abs(fit.defence[i]) < 0.05
