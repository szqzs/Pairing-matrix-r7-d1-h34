import pytest

from rank7_jk.config import FormulaConfig
from rank7_jk.invariants import (
    InvariantMonomial,
    InvariantParseError,
    class_ranks,
    gamma_labels,
)


def test_rank7_label_sets_have_expected_order():
    config = FormulaConfig()

    assert class_ranks(config.rank) == (2, 3, 4, 5, 6, 7)
    assert gamma_labels(config.rank)[:4] == ((2, 2), (2, 3), (2, 4), (2, 5))
    assert gamma_labels(config.rank)[-1] == (7, 7)
    assert len(gamma_labels(config.rank)) == 21


def test_parse_identity_and_canonical_roundtrip():
    config = FormulaConfig(rank=5, genus=2)

    identity = InvariantMonomial.from_string(config, "1")
    assert identity.is_identity
    assert str(identity) == "1"
    assert InvariantMonomial.from_dict(identity.to_dict()) == identity

    monomial = InvariantMonomial.from_string(config, "f2^8 * gamma(2,2) a3")
    assert str(monomial) == "a3 f2^8 gamma22"
    assert InvariantMonomial.from_string(config, str(monomial)) == monomial


def test_symmetric_gamma_parser_normalizes_label_order():
    config = FormulaConfig(rank=5, genus=2)

    assert (
        InvariantMonomial.from_string(config, "gamma32")
        == InvariantMonomial.from_string(config, "gamma23")
    )
    assert (
        InvariantMonomial.from_string(config, "gamma2_4")
        == InvariantMonomial.from_string(config, "gamma(2,4)")
    )


@pytest.mark.parametrize(
    ("text", "ordinary_degree", "chern_degree"),
    [
        ("f2^11", 22, 22),
        ("f2^13", 26, 26),
        ("a2 f2^9", 22, 20),
        ("f2^8 gamma22", 22, 20),
        ("f2^7 f3^2", 22, 20),
        ("a2^2 f2^9", 26, 22),
    ],
)
def test_rank5_public_degree_conventions(text, ordinary_degree, chern_degree):
    config = FormulaConfig(rank=5, genus=2)
    monomial = InvariantMonomial.from_string(config, text)

    assert monomial.ordinary_degree == ordinary_degree
    assert monomial.chern_degree == chern_degree


def test_monomial_multiplication_adds_exponents_and_degrees():
    config = FormulaConfig(rank=5, genus=2)
    left = InvariantMonomial.from_string(config, "a2 f2^9")
    right = InvariantMonomial.from_string(config, "f2^13")
    product = left * right

    assert str(product) == "a2 f2^22"
    assert product.ordinary_degree == left.ordinary_degree + right.ordinary_degree
    assert product.chern_degree == left.chern_degree + right.chern_degree


@pytest.mark.parametrize(
    "text",
    [
        "a1",
        "f6",
        "gamma18",
        "gamma222",
        "gamma(2)",
        "f2^-1",
    ],
)
def test_parser_rejects_invalid_rank5_factors(text):
    config = FormulaConfig(rank=5, genus=2)

    with pytest.raises(InvariantParseError):
        InvariantMonomial.from_string(config, text)


def test_monomials_from_different_ranks_do_not_multiply():
    left = InvariantMonomial.from_string(FormulaConfig(rank=5, genus=2), "f2")
    right = InvariantMonomial.from_string(FormulaConfig(rank=7, genus=2), "f2")

    with pytest.raises(ValueError):
        left * right
