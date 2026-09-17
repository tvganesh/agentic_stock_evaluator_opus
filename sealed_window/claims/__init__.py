"""Claims: the typed, falsifiable unit that models are allowed to produce.

An agent may not return a number representing its opinion. It returns claims, each with
evidence IDs and a *falsifier* -- a predicate over snapshot columns which, if true, refutes
the claim. The veto auditor returns refutations, which carry falsifiers too.

Modules
-------
schema     Pydantic models: what a model may emit (drafts) and what the harness records.
dsl        The restricted falsifier language: tokenizer, parser, evaluator, reachability.
validator  Parse-time checks run before adjudication (evidence, subject, columns, figures).
"""
