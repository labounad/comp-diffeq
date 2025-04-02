# CLI helpers

from sympy import symbols, lambdify, sympify


def parse_function_string(func_str):
    x = symbols('x')
    expr = sympify(func_str)
    return lambdify(x, expr, "numpy")
