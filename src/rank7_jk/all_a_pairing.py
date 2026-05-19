"""Specialized all-a pairing evaluator for the one-defect c18 probe.

This is not the full production evaluator.  It implements the narrow formula
needed when the H62 test column has only ``a`` classes and the H34 c18 source
row has exactly one defect: either one ``f_r`` or one ``gamma_rs``.
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial
from typing import Sequence, Tuple

from . import modular_formula
from .c18_basis import C18SourceRow, H62TestColumn
from .config import FormulaConfig, RANK7_G2_D1
from .exterior import ExteriorAlgebra
from .invariants import InvariantMonomial
from .mod_arith import require_prime
from .residue_transition import residue_poly_mod
from .sparse_poly import SparsePoly, add, clean, constant, mul, pow_poly

DerivOrders = Tuple[int, ...]


def c18_all_a_pairing_column(
    index: int,
    column: H62TestColumn,
    rows: Sequence[C18SourceRow],
    prime: int,
    *,
    config: FormulaConfig = RANK7_G2_D1,
) -> Tuple[int, ...]:
    """Evaluate one actual all-a probe column against all c18 source rows."""

    del index
    if column.kind != "all_a":
        raise ValueError("c18_all_a_pairing_column expects an all-a test column")
    return tuple(
        all_a_pairing_value(config, row.monomial, column.monomial, prime=prime)
        for row in rows
    )


def all_a_pairing_value(
    config: FormulaConfig,
    source: InvariantMonomial,
    test: InvariantMonomial,
    *,
    prime: int | None = None,
) -> int:
    """Evaluate a one-defect source monomial against an all-a test monomial."""

    if any(test.f_exp) or any(test.gamma_exp):
        raise ValueError("the specialized all-a evaluator requires an all-a test")
    return all_a_pairing_total_mod(config, source * test, prime=prime)


def all_a_pairing_total_mod(
    config: FormulaConfig,
    total: InvariantMonomial,
    *,
    prime: int | None = None,
) -> int:
    """Evaluate a top-degree all-a times one-defect total monomial modulo a prime."""

    p = require_prime(config.primary_prime if prime is None else prime)
    _validate_total(config, total)
    target_delta = tuple(int(exp) for exp in total.f_exp[1:])
    gamma_exp = tuple(int(exp) for exp in total.gamma_exp)

    shared_terms = _shared_kernel_terms(config, target_delta, gamma_exp, p)
    if not shared_terms:
        return 0

    a_poly = dict(_tau_power_mod(config, tuple(total.a_exp), p))
    value = 0
    for deriv_orders, shared_items in shared_terms:
        full_poly = mul(a_poly, dict(shared_items), prime=p)
        if full_poly:
            value = (
                value
                + residue_poly_mod(
                    config.rank,
                    full_poly,
                    deriv_orders,
                    prime=p,
                    root_power=config.root_denominator_power,
                )
            ) % p

    scale_factor = int(config.collapsed_prefactor) % p
    for exp in total.f_exp:
        scale_factor = scale_factor * factorial(int(exp)) % p
    return scale_factor * value % p


def _validate_total(config: FormulaConfig, total: InvariantMonomial) -> None:
    if total.rank != config.rank:
        raise ValueError("pairing monomial rank must match the formula config")
    if total.ordinary_degree != config.top_degree:
        raise ValueError("pairing monomial must have top ordinary degree")
    if sum(total.f_exp) + sum(total.gamma_exp) != 1:
        raise NotImplementedError(
            "the all-a specialized evaluator currently supports exactly one defect"
        )
    if sum(total.f_exp[1:]) > 1:
        raise NotImplementedError("at most one non-f2 delta defect is supported")


@lru_cache(maxsize=None)
def _shared_kernel_terms(
    config: FormulaConfig,
    target_delta: Tuple[int, ...],
    gamma_exp: Tuple[int, ...],
    prime: int,
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    if len(target_delta) != len(config.delta_ranks):
        raise ValueError("target delta length does not match the formula config")
    if len(gamma_exp) != len(config.gamma_labels):
        raise ValueError("gamma exponent length does not match the formula config")

    gamma_count = sum(gamma_exp)
    if gamma_count == 0:
        return _even_kernel_terms(config, target_delta, p)
    if gamma_count == 1 and not any(target_delta):
        return _gamma_zero_delta_terms(config, gamma_exp, p)
    raise NotImplementedError(
        "the first all-a evaluator supports either one f defect or one gamma defect"
    )


@lru_cache(maxsize=None)
def _even_kernel_terms(
    config: FormulaConfig,
    target_delta: Tuple[int, ...],
    prime: int,
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    zero_deriv = _zero_deriv(config)
    if not any(target_delta):
        unit = tuple(sorted(constant(config.y_count, 1, prime=p).items()))
        return ((zero_deriv, unit),)
    if sum(target_delta) != 1:
        raise NotImplementedError("only zero or one delta defect is supported")

    delta_rank = config.delta_ranks[target_delta.index(1)]
    poly0 = add(
        dict(modular_formula.c_tilde_delta_coeff_mod(config, delta_rank, p)),
        dict(_det_ratio_first_delta_mod(config, delta_rank, p)),
        prime=p,
        scale=config.genus,
    )

    terms = []
    if poly0:
        terms.append((zero_deriv, tuple(sorted(poly0.items()))))
    for j in range(1, config.y_count + 1):
        deriv = [0 for _ in range(config.y_count)]
        deriv[j - 1] = 1
        poly = dict(modular_formula.b_perturbation_mod(config, delta_rank, j, p))
        if poly:
            terms.append((tuple(deriv), tuple(sorted(clean(poly, p).items()))))
    return tuple(terms)


@lru_cache(maxsize=None)
def _gamma_zero_delta_terms(
    config: FormulaConfig,
    gamma_exp: Tuple[int, ...],
    prime: int,
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    exterior = ExteriorAlgebra(config)
    acc: SparsePoly = {}
    for mask, coeff in exterior.gamma_product_to_mask_poly(gamma_exp).items():
        term = dict(_b_hat_mask_delta_zero_mod(config, mask, p))
        if term:
            acc = add(acc, term, prime=p, scale=coeff)
    if not acc:
        return ()
    return ((_zero_deriv(config), tuple(sorted(acc.items()))),)


@lru_cache(maxsize=None)
def _b_hat_mask_delta_zero_mod(
    config: FormulaConfig,
    mask: int,
    prime: int,
) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    p = require_prime(prime)
    exterior = ExteriorAlgebra(config)
    target = exterior.labels_from_mask(int(mask))
    if len(target) % 2:
        return ()
    if not target:
        return tuple(sorted(constant(config.y_count, 1, prime=p).items()))
    if len(target) != 2:
        raise NotImplementedError("only one gamma factor is supported in the all-a probe")

    acc: SparsePoly = {}
    for i in range(1, config.genus + 1):
        left_side = i
        right_side = i + config.genus
        left_labels = [label for label in target if label[1] == left_side]
        right_labels = [label for label in target if label[1] == right_side]
        for left_label in left_labels:
            for right_label in right_labels:
                wedge = exterior.wedge_masks(
                    exterior.mask_for_b_label(left_label),
                    exterior.mask_for_b_label(right_label),
                )
                if wedge is None:
                    continue
                sign, pair_mask = wedge
                if pair_mask != mask:
                    continue
                pair_poly = dict(
                    modular_formula.hat_pair_delta_zero_mod(
                        config,
                        left_label[0],
                        right_label[0],
                        p,
                    )
                )
                acc = add(acc, pair_poly, prime=p, scale=sign)
    return tuple(sorted(acc.items()))


@lru_cache(maxsize=None)
def _det_ratio_first_delta_mod(
    config: FormulaConfig,
    delta_rank: int,
    prime: int,
) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    p = require_prime(prime)
    h0_inv = modular_formula.hessian_tau2_inverse_mod(config, p)
    h_delta = modular_formula.tau_hessian_mod(config, delta_rank, p)
    acc: SparsePoly = {}
    for i in range(config.y_count):
        for k in range(config.y_count):
            coeff = h0_inv[i][k] % p
            if coeff:
                acc = add(acc, dict(h_delta[k][i]), prime=p, scale=coeff)
    return tuple(sorted(clean(acc, p).items()))


@lru_cache(maxsize=None)
def _tau_power_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    prime: int,
) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    p = require_prime(prime)
    if len(a_exp) != len(config.class_ranks):
        raise ValueError("a exponent length does not match the formula config")
    out = constant(config.y_count, 1, prime=p)
    for exp, r in zip(a_exp, config.class_ranks):
        if exp:
            factor = dict(modular_formula.tau_mod(config, r, p))
            out = mul(out, pow_poly(factor, int(exp), prime=p), prime=p)
    return tuple(sorted(out.items()))


def _zero_deriv(config: FormulaConfig) -> DerivOrders:
    return tuple(0 for _ in range(config.y_count))
