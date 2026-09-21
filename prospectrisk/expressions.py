"""Safe, vectorised formula evaluation for dependent parameters.

A dependent parameter is computed from other inputs of the same segment, e.g.

    hc_saturation = 1 - 0.08 / porosity
    porosity      = 0.42 * exp(-0.00035 * contact_depth)
    rf_solution_gas = rf_oil

Only arithmetic, comparisons, numeric constants, input names and a whitelist of
NumPy functions are accepted; the expression is parsed with :mod:`ast` and never
passed to ``eval``.
"""
from __future__ import annotations

import ast
from typing import Callable

import numpy as np


class ExpressionError(ValueError):
    pass


FUNCTIONS: dict[str, Callable] = {
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs,
    "min": np.minimum, "max": np.maximum, "clip": np.clip, "where": np.where,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
}
CONSTANTS = {"pi": float(np.pi), "e": float(np.e)}

_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
_UNARY = {ast.USub: np.negative, ast.UAdd: np.positive}
_CMP = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal}


def parse(expr: str) -> tuple[ast.Expression, set[str]]:
    """Parse and validate; returns the tree and the set of variable names referenced."""
    if not isinstance(expr, str) or not expr.strip():
        raise ExpressionError("Formula is empty")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Formula syntax error: {exc.msg}") from exc
    names: set[str] = set()

    def visit(node):
        if isinstance(node, ast.Expression):
            visit(node.body)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN:
                raise ExpressionError(f"Operator '{type(node.op).__name__}' is not allowed")
            visit(node.left)
            visit(node.right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY:
                raise ExpressionError(f"Operator '{type(node.op).__name__}' is not allowed")
            visit(node.operand)
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or type(node.ops[0]) not in _CMP:
                raise ExpressionError("Only single comparisons with <, <=, >, >= are allowed")
            visit(node.left)
            visit(node.comparators[0])
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExpressionError("Only numeric constants are allowed")
        elif isinstance(node, ast.Name):
            if node.id not in FUNCTIONS and node.id not in CONSTANTS:
                names.add(node.id)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise ExpressionError("Unknown function; allowed: " + ", ".join(sorted(FUNCTIONS)))
            if node.keywords:
                raise ExpressionError("Keyword arguments are not allowed")
            for a in node.args:
                visit(a)
        else:
            raise ExpressionError(f"'{type(node).__name__}' is not allowed in a formula")

    visit(tree)
    return tree, names


def evaluate(tree: ast.Expression, env: dict[str, np.ndarray], n: int) -> np.ndarray:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BinOp):
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Compare):
            return _CMP[type(node.ops[0])](ev(node.left), ev(node.comparators[0]))
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            if node.id not in env:
                raise ExpressionError(f"Unknown input '{node.id}' in formula")
            return env[node.id]
        if isinstance(node, ast.Call):
            fn = FUNCTIONS[node.func.id]
            try:
                return fn(*[ev(a) for a in node.args])
            except TypeError as exc:
                raise ExpressionError(f"{node.func.id}(): {exc}") from exc
        raise ExpressionError("Invalid formula")  # pragma: no cover - parse() prevents this

    with np.errstate(all="ignore"):
        out = np.asarray(ev(tree), dtype=float)
    return np.broadcast_to(out, (n,)).astype(float)
