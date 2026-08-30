"""Static checks on the page template.

These exist because of a real bug: the Phase 2 rewrite deleted two elements but
left the script still setting their textContent. The uncaught TypeError killed
every section after it - score matrix, ratings, fixtures, the whole validation
block - on a page that still looked fine at a glance and passed a syntax check.
Syntax checking a script does not tell you the DOM it talks to still exists.
"""
import re
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / "web" / "template.html"


@pytest.fixture(scope="module")
def html():
    return TEMPLATE.read_text(encoding="utf-8")


def test_every_referenced_id_exists(html):
    declared = set(re.findall(r'id="([\w-]+)"', html))
    referenced = set(re.findall(r'getElementById\(["\']([\w-]+)["\']\)', html))
    missing = referenced - declared
    assert not missing, f"script references ids that no element declares: {sorted(missing)}"


def test_no_leftover_placeholder_besides_the_injection_point(html):
    assert html.count("__MODEL_JSON__") == 1, "the model injection point must appear exactly once"


def test_outcome_colour_classes_are_reachable_in_both_contexts(html):
    """The bh/bd/ba classes are used in two places; a selector scoped to only one
    leaves the other painting white text on no background."""
    for cls in ("bh", "bd", "ba"):
        rule = re.search(rf"([^{{}}]*\.{cls})\{{background:var\(--\w+\)\}}", html)
        assert rule, f"no background rule found for .{cls}"
        selector = rule.group(1)
        assert ".bar" in selector and ".split" in selector, (
            f".{cls} background is scoped to {selector.strip()!r}; "
            "it must cover both the outcome bar and the fixture splits")


def test_theme_tokens_defined_for_all_three_states(html):
    """Light, prefers-color-scheme dark, and an explicit data-theme dark."""
    assert re.search(r"^:root\{", html, re.M), "no bare :root light palette"
    assert 'prefers-color-scheme:dark' in html
    assert ':root[data-theme="dark"]' in html
    for token in ("--home", "--draw", "--away", "--ink", "--surface"):
        assert html.count(token + ":") >= 3, f"{token} not redefined for every theme state"


def test_body_paints_its_own_background(html):
    """A transparent body borrows the host page's ground and can render one
    theme's text on the other theme's background."""
    body = re.search(r"body\{([^}]*)\}", html)
    assert body and "background:var(--ground)" in body.group(1)
