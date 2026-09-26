"""Tests for the signal table (``claims.signals``) and the claim prompts rendered from it.

The table exists because qwen3:14b, asked to check conditions itself, called 7 of 8 overvalued stocks
"cheap on earnings" and sometimes wrote the falsifier backwards. The table's promise is that each rule's
falsifier is right by construction; these tests are what make that true rather than hoped: every
falsifier is proven to be the exact negation of its rule over sampled values, including the edges, and
every worked example is re-checked against its own rule.
"""

from __future__ import annotations

import random
import re

from sealed_window.agents.prompts import MAX_CLAIMS_PER_CALL, claim_system_prompt, signal_rules_text
from sealed_window.claims import dsl
from sealed_window.claims.signals import NEGATIVE, POSITIVE, SIGNALS, examples_for, rules
from sealed_window.claims.validator import unsupported_figures
from sealed_window.snapshot.columns import COLUMNS, Dimension


def _literals(text: str) -> list[float]:
    """Numeric literals in a predicate, signs included."""
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def test_every_column_is_real_and_in_its_signals_dimension():
    """A misnamed column would make every claim copying that falsifier unevaluable."""
    for signal in SIGNALS:
        for rule in signal.rules:
            for text in (signal.full_condition(rule), rule.falsifier):
                for name in dsl.columns(dsl.parse(text)):
                    assert name in COLUMNS, f"{rule.key}: unknown column {name}"
                    assert COLUMNS[name].dimension is signal.dimension, f"{rule.key}: {name} is another dimension's"


def test_every_falsifier_fits_the_language_limits():
    """Copied verbatim into a claim, the falsifier must pass the same limits a model's own would face."""
    for rule in rules():
        tree = dsl.parse(rule.falsifier)
        assert len(dsl.columns(tree)) <= dsl.MAX_COLUMNS, rule.key
        assert len(dsl.comparisons(tree)) <= dsl.MAX_COMPARISONS, rule.key
        assert len(rule.falsifier) <= dsl.MAX_LENGTH, rule.key


def test_every_falsifier_is_the_exact_negation_of_its_rule():
    """Over sampled rows the falsifier is TRUE exactly when the guarded condition is FALSE.

    Samples cluster on every threshold in either predicate, a hair either side of it, and on equality
    between compared columns, because the errors this guards against -- ``>`` where ``>=`` belongs,
    AND where OR belongs -- live on those edges and nowhere else.
    """
    rng = random.Random(20260926)
    for signal in SIGNALS:
        for rule in signal.rules:
            condition = dsl.parse(signal.full_condition(rule))
            falsifier = dsl.parse(rule.falsifier)
            names = sorted(dsl.columns(condition) | dsl.columns(falsifier))
            edges = set(_literals(signal.full_condition(rule)) + _literals(rule.falsifier)) | {0.0}
            pool = sorted({e + d for e in edges for d in (-1.0, -0.01, 0.0, 0.01, 1.0)} | {-150.0, 150.0})
            numeric = [name for name in names if name != "is_bank"]
            for _ in range(3000):
                row = {name: rng.choice((0.0, 1.0)) if name == "is_bank" else rng.choice(pool) for name in names}
                if len(numeric) > 1 and rng.random() < 0.3:  # equality between compared columns is an edge too
                    a, b = rng.sample(numeric, 2)
                    row[b] = row[a]
                holds = dsl.evaluate(condition, row)
                assert dsl.evaluate(falsifier, row) is (not holds), f"{rule.key} disagrees on {row}"


def test_every_threshold_is_a_round_zone_edge():
    """No falsifier threshold carries measurement-like precision, so none can be pinned to a reading."""
    for rule in rules():
        for value in _literals(rule.falsifier):
            assert round(value, dsl.BOUNDARY_DECIMALS) == value, f"{rule.key}: {value}"


def test_every_signal_has_both_sides_and_examples():
    """A one-sided signal pushes claims one way; a rule without an example teaches nothing."""
    for signal in SIGNALS:
        directions = {rule.direction for rule in signal.rules}
        assert directions == {POSITIVE, NEGATIVE}, f"{signal.key} is one-sided"
        assert sum(len(examples_for(rule)) for rule in signal.rules) >= 4, f"{signal.key} has fewer than 4 examples"
        for rule in signal.rules:
            assert examples_for(rule), f"{rule.key} has no example"


def test_every_example_satisfies_its_rule_and_quotes_only_its_own_figures():
    """An example contradicting its rule would teach exactly the error the table exists to remove."""
    for signal in SIGNALS:
        for rule in signal.rules:
            for example in examples_for(rule):
                row = dict(example.values)
                assert dsl.evaluate(dsl.parse(signal.full_condition(rule)), row), f"{rule.key}: {example.symbol}"
                assert not dsl.evaluate(dsl.parse(rule.falsifier), row), f"{rule.key}: {example.symbol}"
                statement = rule.render(example)
                assert unsupported_figures(statement, [{"fields": row}]) == [], statement
                assert statement.index(example.symbol) == 0, "figures and conclusion follow the company name"


def test_prompt_carries_every_rule_and_falsifier_for_its_own_dimension_only():
    """Each analyst sees its own table, with the claim limit the schema enforces; news sees none."""
    for dimension in (Dimension.FUNDAMENTAL, Dimension.TECHNICAL):
        prompt = claim_system_prompt(dimension)
        assert signal_rules_text(dimension) in prompt
        assert f"at most {MAX_CLAIMS_PER_CALL} claims" in prompt
        for signal in SIGNALS:
            for rule in signal.rules:
                present = f"Falsifier: {rule.falsifier}" in prompt
                assert present is (signal.dimension is dimension), f"{rule.key} in the {dimension.value} prompt"
    assert "Signal rules" not in claim_system_prompt(Dimension.NEWS)


def test_rsi_zones_and_macd_lines_are_spelled_out():
    """The zones agreed on 26 Sep 2026, and which MACD line is which, reach the technical analyst."""
    text = signal_rules_text(Dimension.TECHNICAL)
    for zone in ("at or below 30 oversold", "30-45 weak", "45-55 neutral", "55-70 bullish", "at or above 70 overbought"):
        assert zone in text
    assert "the MACD line is above the signal line" in text
    assert "the signal line is above the MACD line" in text
