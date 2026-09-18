"""The falsifier DSL: a small, auditable predicate language over snapshot columns.

Open decision 2 in ARCHITECTURE_OPUS.md recommends a restricted DSL over restricted Python;
this is it. Nothing here calls ``eval``, touches attributes, or can loop. Grammar::

    expr     := or_expr
    or_expr  := and_expr ("OR" and_expr)*
    and_expr := not_expr ("AND" not_expr)*
    not_expr := "NOT" not_expr | "(" expr ")" | comparison
    compare  := arith ("<" | "<=" | ">" | ">=" | "==" | "!=") arith
    arith    := term (("+" | "-") term)*
    term     := factor (("*" | "/") factor)*
    factor   := NUMBER | COLUMN | "-" factor | FUNC "(" arith ("," arith)* ")" | "(" arith ")"
    FUNC     := abs | min | max

Keywords are case-insensitive; columns must be names from ``snapshot.columns.COLUMNS``.
Hard limits (length, tokens, nesting depth, comparisons, distinct columns) keep every
falsifier small enough for a human to read in the dossier.

Semantics used by the adjudicator:

* :func:`evaluate` returns True if the falsifier *fires* (refutes its claim). A ``None``
  column value or a division by zero raises :class:`Unevaluable` -- unevaluable is not true.
* :func:`is_reachable` is the vacuity check: can this falsifier fire on values actually
  observed across the snapshot's universe? A falsifier like ``close < 0`` can never fire,
  so a claim carrying it would be unkillable -- exactly what a prompt injection would ask
  for. Such claims are discarded as VACUOUS.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Mapping, Sequence, Union

from ..snapshot.columns import COLUMNS

MAX_LENGTH = 300
MAX_TOKENS = 80
MAX_DEPTH = 8
MAX_COMPARISONS = 6
MAX_COLUMNS = 4


class FalsifierError(ValueError):
    """The falsifier text is not a valid DSL expression (syntax, unknown column or limits)."""


class Unevaluable(Exception):
    """The falsifier is valid but cannot be evaluated against this row (missing data, div by zero)."""


# ---- AST -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Num:
    """Numeric literal."""

    value: float


@dataclass(frozen=True)
class Col:
    """Reference to a snapshot column."""

    name: str


@dataclass(frozen=True)
class Neg:
    """Unary minus."""

    operand: "Arith"


@dataclass(frozen=True)
class BinOp:
    """Binary arithmetic: + - * /."""

    op: str
    left: "Arith"
    right: "Arith"


@dataclass(frozen=True)
class Func:
    """Whitelisted function call: abs, min, max."""

    name: str
    args: tuple["Arith", ...]


@dataclass(frozen=True)
class Compare:
    """Comparison between two arithmetic expressions."""

    op: str
    left: "Arith"
    right: "Arith"


@dataclass(frozen=True)
class BoolOp:
    """N-ary AND / OR."""

    op: str
    items: tuple["BoolExpr", ...]


@dataclass(frozen=True)
class Not:
    """Logical negation."""

    operand: "BoolExpr"


Arith = Union[Num, Col, Neg, BinOp, Func]
BoolExpr = Union[Compare, BoolOp, Not]

# ---- tokenizer -----------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)|(?P<op><=|>=|==|!=|[<>()+\-*/,]))"
)
_FUNCS = {"abs", "min", "max"}
_KEYWORDS = {"and", "or", "not"}
_CMP_OPS = {"<", "<=", ">", ">=", "==", "!="}


def tokenize(text: str) -> list[tuple[str, str]]:
    """Split falsifier text into ``(kind, value)`` tokens, enforcing length and token limits."""
    if len(text) > MAX_LENGTH:
        raise FalsifierError(f"falsifier longer than {MAX_LENGTH} characters")
    tokens: list[tuple[str, str]] = []
    pos = 0
    stripped = text.rstrip()
    while pos < len(stripped):
        match = _TOKEN_RE.match(stripped, pos)
        if not match or match.end() == pos:
            raise FalsifierError(f"unexpected character at position {pos}: {stripped[pos:pos + 10]!r}")
        pos = match.end()
        if match.group("num") is not None:
            tokens.append(("num", match.group("num")))
        elif match.group("ident") is not None:
            ident = match.group("ident")
            lowered = ident.lower()
            if lowered in _KEYWORDS:
                tokens.append(("kw", lowered))
            elif lowered in _FUNCS:
                tokens.append(("func", lowered))
            else:
                tokens.append(("col", ident))
        else:
            tokens.append(("op", match.group("op")))
    if not tokens:
        raise FalsifierError("empty falsifier")
    if len(tokens) > MAX_TOKENS:
        raise FalsifierError(f"falsifier has more than {MAX_TOKENS} tokens")
    return tokens


# ---- parser --------------------------------------------------------------------------


class _Parser:
    """Recursive-descent parser with bounded backtracking for parenthesised groups."""

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        """Start at the first token."""
        self.tokens = tokens
        self.pos = 0
        self.depth = 0

    def peek(self, offset: int = 0) -> tuple[str, str] | None:
        """Return the token ``offset`` ahead without consuming it."""
        index = self.pos + offset
        return self.tokens[index] if index < len(self.tokens) else None

    def take(self, kind: str | None = None, value: str | None = None) -> tuple[str, str]:
        """Consume the next token, requiring ``kind``/``value`` if given."""
        token = self.peek()
        if token is None or (kind and token[0] != kind) or (value and token[1] != value):
            raise FalsifierError(f"expected {value or kind} but found {token[1] if token else 'end of input'}")
        self.pos += 1
        return token

    def _enter(self) -> None:
        """Track nesting depth and enforce :data:`MAX_DEPTH`."""
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise FalsifierError(f"nesting deeper than {MAX_DEPTH}")

    def parse(self) -> BoolExpr:
        """Parse the full token stream as one boolean expression."""
        expr = self.or_expr()
        if self.peek() is not None:
            raise FalsifierError(f"unexpected trailing token {self.peek()[1]!r}")
        return expr

    def or_expr(self) -> BoolExpr:
        """``and_expr ("OR" and_expr)*``"""
        items = [self.and_expr()]
        while self.peek() == ("kw", "or"):
            self.take()
            items.append(self.and_expr())
        return items[0] if len(items) == 1 else BoolOp("or", tuple(items))

    def and_expr(self) -> BoolExpr:
        """``not_expr ("AND" not_expr)*``"""
        items = [self.not_expr()]
        while self.peek() == ("kw", "and"):
            self.take()
            items.append(self.not_expr())
        return items[0] if len(items) == 1 else BoolOp("and", tuple(items))

    def not_expr(self) -> BoolExpr:
        """``"NOT" not_expr | "(" expr ")" | comparison`` with backtracking on "("."""
        self._enter()
        try:
            if self.peek() == ("kw", "not"):
                self.take()
                return Not(self.not_expr())
            if self.peek() == ("op", "("):
                saved = self.pos
                try:
                    return self.comparison()
                except FalsifierError:
                    self.pos = saved
                self.take("op", "(")
                inner = self.or_expr()
                self.take("op", ")")
                return inner
            return self.comparison()
        finally:
            self.depth -= 1

    def comparison(self) -> Compare:
        """``arith CMP arith``"""
        left = self.arith()
        token = self.peek()
        if token is None or token[0] != "op" or token[1] not in _CMP_OPS:
            raise FalsifierError("expected a comparison operator")
        self.take()
        return Compare(token[1], left, self.arith())

    def arith(self) -> Arith:
        """``term (("+" | "-") term)*``"""
        node = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            node = BinOp(op, node, self.term())
        return node

    def term(self) -> Arith:
        """``factor (("*" | "/") factor)*``"""
        node = self.factor()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            node = BinOp(op, node, self.factor())
        return node

    def factor(self) -> Arith:
        """Literal, column, unary minus, function call or parenthesised arithmetic."""
        self._enter()
        try:
            token = self.peek()
            if token is None:
                raise FalsifierError("unexpected end of input")
            kind, value = token
            if kind == "num":
                self.take()
                return Num(float(value))
            if kind == "col":
                self.take()
                if value not in COLUMNS:
                    raise FalsifierError(f"unknown column {value!r}")
                return Col(value)
            if token == ("op", "-"):
                self.take()
                return Neg(self.factor())
            if kind == "func":
                self.take()
                self.take("op", "(")
                args = [self.arith()]
                while self.peek() == ("op", ","):
                    self.take()
                    args.append(self.arith())
                self.take("op", ")")
                if value == "abs" and len(args) != 1:
                    raise FalsifierError("abs takes exactly one argument")
                if value in ("min", "max") and len(args) < 2:
                    raise FalsifierError(f"{value} takes at least two arguments")
                return Func(value, tuple(args))
            if token == ("op", "("):
                self.take()
                inner = self.arith()
                self.take("op", ")")
                return inner
            raise FalsifierError(f"unexpected token {value!r}")
        finally:
            self.depth -= 1


def parse(text: str) -> BoolExpr:
    """Parse falsifier text into an AST, enforcing grammar, known columns and all limits."""
    tree = _Parser(tokenize(text)).parse()
    if len(comparisons(tree)) > MAX_COMPARISONS:
        raise FalsifierError(f"more than {MAX_COMPARISONS} comparisons")
    if len(columns(tree)) > MAX_COLUMNS:
        raise FalsifierError(f"more than {MAX_COLUMNS} distinct columns")
    if not columns(tree):
        raise FalsifierError("falsifier references no snapshot column")
    return tree


# ---- analysis ------------------------------------------------------------------------


def _walk(node) -> Sequence:
    """Yield every node in the tree (pre-order)."""
    stack, out = [node], []
    while stack:
        current = stack.pop()
        out.append(current)
        if isinstance(current, (BinOp, Compare)):
            stack.extend([current.right, current.left])
        elif isinstance(current, (Neg, Not)):
            stack.append(current.operand)
        elif isinstance(current, Func):
            stack.extend(reversed(current.args))
        elif isinstance(current, BoolOp):
            stack.extend(reversed(current.items))
    return out


def columns(tree: BoolExpr) -> set[str]:
    """Distinct column names referenced by a falsifier (used for SLA and dimension checks)."""
    return {node.name for node in _walk(tree) if isinstance(node, Col)}


def comparisons(tree: BoolExpr) -> list[Compare]:
    """All comparison nodes in a falsifier."""
    return [node for node in _walk(tree) if isinstance(node, Compare)]


# ---- evaluation ----------------------------------------------------------------------


def _arith(node: Arith, row: Mapping[str, float | None]) -> float:
    """Evaluate an arithmetic node against a row; missing values raise :class:`Unevaluable`."""
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Col):
        value = row.get(node.name)
        if value is None:
            raise Unevaluable(f"column {node.name} has no value")
        return float(value)
    if isinstance(node, Neg):
        return -_arith(node.operand, row)
    if isinstance(node, BinOp):
        left, right = _arith(node.left, row), _arith(node.right, row)
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if right == 0:
            raise Unevaluable("division by zero")
        return left / right
    if isinstance(node, Func):
        args = [_arith(arg, row) for arg in node.args]
        return abs(args[0]) if node.name == "abs" else (min(args) if node.name == "min" else max(args))
    raise Unevaluable(f"unsupported node {type(node).__name__}")


def evaluate(tree: BoolExpr, row: Mapping[str, float | None]) -> bool:
    """Return True if the falsifier fires on ``row`` (i.e. its claim is refuted)."""
    if isinstance(tree, Compare):
        left, right = _arith(tree.left, row), _arith(tree.right, row)
        return {
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
            "==": left == right,
            "!=": left != right,
        }[tree.op]
    if isinstance(tree, BoolOp):
        results = [evaluate(item, row) for item in tree.items]
        return all(results) if tree.op == "and" else any(results)
    if isinstance(tree, Not):
        return not evaluate(tree.operand, row)
    raise Unevaluable(f"unsupported node {type(tree).__name__}")


STRICT_OPERATORS = frozenset({"<", ">"})
"""Only strict comparisons can be pinned: ``x >= v`` fires when ``x == v``, so it is a real test."""

BOUNDARY_DECIMALS = 2
"""A threshold carrying more precision than this was copied from a measurement, not chosen.

Thresholds a person picks are round -- 0, 30, 1.5, 100. A threshold like 34.661507 matching the
subject's RSI to six places was read off the row. The guard matters: ``news_count_7d > 0`` against
a count of 0 is pinned *and legitimate* (the claim is "no headlines"; one headline disproves it),
so precision, not zero margin, is what separates a copied value from a natural boundary."""


def _literal(node: Arith) -> float | None:
    """Constant value of an arithmetic node, unwrapping unary minus; None if it references a column.

    The tokenizer matches only unsigned numerals, so ``> -1.6`` parses as ``Neg(Num(1.6))``. Without
    unwrapping here, every negative threshold -- which is most pinned technical falsifiers -- is missed.
    """
    if isinstance(node, Num):
        return float(node.value)
    if isinstance(node, Neg):
        inner = _literal(node.operand)
        return None if inner is None else -inner
    return None


def pinned_comparisons(tree: BoolExpr, row: Mapping[str, float | None]) -> list[Compare]:
    """Comparisons whose threshold is the subject's own measured value, so they can never fire.

    The loophole this closes: a model reads ``rsi_14 = 34.661507`` off the evidence and writes
    ``rsi_14 > 34.661507`` as its disproof condition. The claim then passes every other check --
    the predicate is well-formed, evaluable, and reachable across the universe, since other
    companies do sit above that level -- while failing by exactly zero for the one company it
    describes. :func:`is_reachable` cannot catch it, because it asks whether the condition could
    fire for *some* company, not for *this* one.
    """
    pinned = []
    for node in comparisons(tree):
        if node.op not in STRICT_OPERATORS:
            continue
        for column_side, literal_side in ((node.left, node.right), (node.right, node.left)):
            if not isinstance(column_side, Col):
                continue
            threshold = _literal(literal_side)
            observed = row.get(column_side.name)
            if threshold is None or observed is None:
                continue
            if float(observed) == threshold and round(threshold, BOUNDARY_DECIMALS) != threshold:
                pinned.append(node)
                break
    return pinned


def is_reachable(
    tree: BoolExpr,
    row: Mapping[str, float | None],
    scenario_values: Mapping[str, Sequence[float]],
) -> bool:
    """Vacuity check: True if the falsifier fires for some plausible combination of column values.

    ``scenario_values[column]`` holds values observed across the universe (the adjudicator
    passes the 5th/25th/50th/75th/95th percentiles). Each referenced column ranges over its
    scenario values plus the subject's own value; the others stay at the subject's values.
    """
    names = sorted(columns(tree))
    choices = []
    for name in names:
        values = list(scenario_values.get(name, ()))
        if row.get(name) is not None:
            values.append(float(row[name]))
        if not values:
            return False
        choices.append(sorted(set(values)))
    for combination in itertools.product(*choices):
        trial = dict(row)
        trial.update(zip(names, combination))
        try:
            if evaluate(tree, trial):
                return True
        except Unevaluable:
            continue
    return False
