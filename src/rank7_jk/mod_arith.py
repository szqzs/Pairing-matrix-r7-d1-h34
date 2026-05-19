"""Small modular-arithmetic helpers for fast JK kernels."""

from __future__ import annotations

import sympy as sp


def require_prime(prime: int) -> int:
    p = int(prime)
    if p <= 1 or not sp.isprime(p):
        raise ValueError("prime must be prime")
    return p


def normalize(value: int, prime: int) -> int:
    return int(value) % int(prime)


def mod_inv(value: int, prime: int) -> int:
    p = require_prime(prime)
    value = int(value) % p
    if value == 0:
        raise ZeroDivisionError(f"denominator is 0 modulo prime {p}")
    return pow(value, p - 2, p)


def rational_mod(value: sp.Expr, prime: int) -> int:
    p = require_prime(prime)
    rational = sp.Rational(value)
    denominator = int(rational.q) % p
    if denominator == 0:
        raise ZeroDivisionError(f"rational denominator is 0 modulo prime {p}")
    return int(rational.p) % p * mod_inv(denominator, p) % p
