"""Restricted external scalar grammar; never evaluate Python source text."""

import ast
import keyword
import math
import operator

from sympy import E, Expr, Float, Integer, Symbol, cos, exp, log, pi, sin, sqrt, tan


FUNCTIONS = {"sin": sin, "cos": cos, "tan": tan, "sqrt": sqrt, "exp": exp, "log": log}
CONSTANTS = {"pi": pi, "E": E}
OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}


def validate_variable(variable: str) -> None:
    if (not isinstance(variable, str) or not variable.isidentifier()
            or keyword.iskeyword(variable) or variable in FUNCTIONS or variable in CONSTANTS):
        raise ValueError("Variable must be an identifier distinct from functions and constants.")


def parse_scalar(text: str, variable: str) -> Expr:
    """Validate the entire AST before constructing any SymPy expression."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Component expression must be nonempty text.")
    validate_variable(variable)
    try:
        tree = ast.parse(text, mode="eval").body
        _validate(tree, variable)
        return _build(tree, variable)
    except (SyntaxError, TypeError, ArithmeticError, RecursionError) as error:
        raise ValueError(f"Invalid component expression: {text}") from error


def _validate(node: ast.AST, variable: str) -> None:
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Only real numeric literals are allowed.")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise ValueError("Numeric literals must be finite.")
    elif isinstance(node, ast.Name):
        if node.id not in {variable, *CONSTANTS}:
            raise ValueError(f"Unsupported symbol: {node.id}")
    elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        _validate(node.left, variable)
        _validate(node.right, variable)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _validate(node.operand, variable)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in FUNCTIONS or node.keywords:
            raise ValueError("Only supported functions with positional arguments are allowed.")
        arities = (1, 2) if name == "log" else (1,)
        if len(node.args) not in arities:
            raise ValueError(f"Invalid number of arguments for {name}.")
        for argument in node.args:
            _validate(argument, variable)
    else:
        raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")


def _build(node: ast.AST, variable: str) -> Expr:
    if isinstance(node, ast.Constant):
        return Integer(node.value) if type(node.value) is int else Float(node.value)
    if isinstance(node, ast.Name):
        return Symbol(variable) if node.id == variable else CONSTANTS[node.id]
    if isinstance(node, ast.BinOp):
        return OPERATORS[type(node.op)](_build(node.left, variable), _build(node.right, variable))
    if isinstance(node, ast.UnaryOp):
        value = _build(node.operand, variable)
        return -value if isinstance(node.op, ast.USub) else value
    return FUNCTIONS[node.func.id](*(_build(arg, variable) for arg in node.args))
