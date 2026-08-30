"""Runtime paths and tunables, all overridable by environment."""
from __future__ import annotations

import os
from pathlib import Path

DB_PATH = Path(os.environ.get("HYPZ_DB", "/data/db/sim.db"))
RAW_DIR = Path(os.environ.get("HYPZ_RAW", "/data/raw"))

# Dixon-Coles time decay. Matches this many days old carry half the weight of
# today's. 270 is a middle course: short enough to track form and transfers,
# long enough that a promoted side isn't fitted on twenty games alone.
HALF_LIFE_DAYS = float(os.environ.get("HYPZ_HALF_LIFE_DAYS", "270"))

# Scoreline grid for the closed-form bivariate Poisson. Ten goals is far into
# the tail for league football; truncating there costs <0.01% of the mass.
MAX_GOALS = 10
