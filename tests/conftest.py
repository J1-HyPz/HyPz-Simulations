import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Point the package at a scratch database before anything imports config.
_TMP = tempfile.mkdtemp(prefix="hypz-test-")
os.environ.setdefault("HYPZ_DB", str(Path(_TMP) / "test.db"))
os.environ.setdefault("HYPZ_RAW", str(Path(_TMP) / "raw"))


@pytest.fixture
def synthetic_matches():
    """A small league with known structure: team A strong, team D weak.

    Synthetic rather than real data so the tests stay hermetic and fast, and so
    the expected ordering is something we control rather than something we look up.
    """
    rng = np.random.default_rng(7)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 0.6, "B": 0.1, "C": -0.1, "D": -0.6}
    start = pd.Timestamp("2020-01-01")
    rows = []
    for week in range(120):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = np.exp(strength[h] - strength[a] + 0.25)
                mu = np.exp(strength[a] - strength[h])
                rows.append({
                    "date": start + pd.Timedelta(days=7 * week),
                    "season": f"{2020 + week // 40}/{(21 + week // 40) % 100:02d}",
                    "home": h, "away": a,
                    "home_score": int(rng.poisson(lam)),
                    "away_score": int(rng.poisson(mu)),
                    "extra_json": "{}",
                })
    return pd.DataFrame(rows)
