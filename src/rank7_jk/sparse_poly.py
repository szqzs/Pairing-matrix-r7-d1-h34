"""Generic sparse polynomials over a prime field.

The production rank-7 evaluator will need sparse arithmetic in six simple-root
variables.  This module is deliberately small: no formula logic, only the
field-polynomial operations that later kernels can share.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

from .mod_arith import require_prime

Alpha = Tuple[int, ...]
SparsePoly = Dict[Alpha, int]


def zero_alpha(var_count: int) -> Alpha:
    if var_count < 0:
        raise ValueError("var_count must be nonnegative")
    return tuple(0 for _ in range(var_count))


def clean(poly: SparsePoly, prime: int) -> SparsePoly:
    p = require_prime(prime)
    return {alpha: coeff % p for alpha, coeff in poly.items() if coeff % p}


def monomial(alpha: Sequence[int], coeff: int = 1, *, prime: int) -> SparsePoly:
    p = require_prime(prime)
    key = tuple(int(item) for item in alpha)
    if any(item < 0 for item in key):
        raise ValueError("monomial exponents must be nonnegative")
    coeff_mod = int(coeff) % p
    return {key: coeff_mod} if coeff_mod else {}


def constant(var_count: int, coeff: int = 1, *, prime: int) -> SparsePoly:
    return monomial(zero_alpha(var_count), coeff, prime=prime)


def add(left: SparsePoly, right: SparsePoly, *, prime: int, scale: int = 1) -> SparsePoly:
    p = require_prime(prime)
    _validate_same_var_count((left, right))
    out = dict(left)
    scale_mod = int(scale) % p
    if not scale_mod:
        return clean(out, p)
    for alpha, coeff in right.items():
        value = (out.get(alpha, 0) + scale_mod * coeff) % p
        if value:
            out[alpha] = value
        else:
            out.pop(alpha, None)
    return clean(out, p)


def scale(poly: SparsePoly, factor: int, *, prime: int) -> SparsePoly:
    p = require_prime(prime)
    factor_mod = int(factor) % p
    if not factor_mod:
        return {}
    return {
        alpha: coeff * factor_mod % p
        for alpha, coeff in poly.items()
        if coeff * factor_mod % p
    }


def _validate_same_var_count(polys: Iterable[SparsePoly]) -> int | None:
    var_count: int | None = None
    for poly in polys:
        for alpha in poly:
            if var_count is None:
                var_count = len(alpha)
            elif len(alpha) != var_count:
                raise ValueError("all sparse polynomial exponents must have the same length")
    return var_count


def mul(left: SparsePoly, right: SparsePoly, *, prime: int) -> SparsePoly:
    p = require_prime(prime)
    if not left or not right:
        return {}
    _validate_same_var_count((left, right))
    out: SparsePoly = {}
    for a1, c1 in left.items():
        for a2, c2 in right.items():
            alpha = tuple(a1[idx] + a2[idx] for idx in range(len(a1)))
            out[alpha] = (out.get(alpha, 0) + c1 * c2) % p
    return clean(out, p)


def pow_poly(base: SparsePoly, exponent: int, *, prime: int) -> SparsePoly:
    p = require_prime(prime)
    exp = int(exponent)
    if exp < 0:
        raise ValueError("polynomial exponent must be nonnegative")
    var_count = _validate_same_var_count((base,))
    if var_count is None:
        raise ValueError("cannot infer variable count from the zero polynomial")
    out = constant(var_count, 1, prime=p)
    cur = clean(base, p)
    while exp:
        if exp & 1:
            out = mul(out, cur, prime=p)
        exp >>= 1
        if exp:
            cur = mul(cur, cur, prime=p)
    return out


def derivative(poly: SparsePoly, var_idx: int, *, prime: int) -> SparsePoly:
    p = require_prime(prime)
    out: SparsePoly = {}
    for alpha, coeff in poly.items():
        idx = int(var_idx)
        if idx < 0 or idx >= len(alpha):
            raise ValueError("var_idx is outside the exponent range")
        power = alpha[idx]
        if not power:
            continue
        next_alpha = list(alpha)
        next_alpha[idx] -= 1
        key = tuple(next_alpha)
        out[key] = (out.get(key, 0) + coeff * power) % p
    return clean(out, p)


def directional_derivative(
    poly: SparsePoly,
    direction: Sequence[int],
    *,
    prime: int,
) -> SparsePoly:
    p = require_prime(prime)
    out: SparsePoly = {}
    for idx, coeff in enumerate(direction):
        if coeff:
            out = add(out, derivative(poly, idx, prime=p), prime=p, scale=int(coeff))
    return clean(out, p)
