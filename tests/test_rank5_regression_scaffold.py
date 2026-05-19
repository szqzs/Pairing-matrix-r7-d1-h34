import pytest

from rank7_jk import slow_evaluator
from rank7_jk.rank5_regression import (
    RANK5_C20_MINOR_FIXTURE,
    RANK5_FORMULA,
    RANK5_PRIMARY_PRIME,
    RANK5_PUBLIC_MINOR_SUMMARIES,
    RANK5_PUBLIC_SCALAR_FIXTURES,
    RANK5_SMALL_PUBLIC_MINOR_FIXTURES,
    RANK5_SOURCE_DEGREE,
    RANK5_TEST_DEGREE,
    RANK5_TOP_DEGREE,
    scalar_fixture_by_name,
)


def test_rank5_formula_and_pairing_degrees_are_frozen():
    assert RANK5_FORMULA.rank == 5
    assert RANK5_FORMULA.genus == 2
    assert RANK5_FORMULA.top_degree == RANK5_TOP_DEGREE == 48
    assert RANK5_SOURCE_DEGREE == 22
    assert RANK5_TEST_DEGREE == 26
    assert RANK5_SOURCE_DEGREE + RANK5_TEST_DEGREE == RANK5_TOP_DEGREE
    assert RANK5_PRIMARY_PRIME == 2305843009213693951


def test_rank5_public_scalar_fixture_values_are_frozen():
    expected = {
        "f2_11__f2_13": 1381783072775710288,
        "f2_9_f3__a2_f2_11": 1438514327499689729,
        "f2_8_f4__f2_13": 513073332518773065,
        "f2_8_gamma22__f2_13": 967147192232714784,
        "f2_7_f3_2__f2_13": 825622484462206102,
    }

    observed = {
        fixture.name: fixture.expected_mod
        for fixture in RANK5_PUBLIC_SCALAR_FIXTURES
    }
    assert observed == expected


def test_rank5_public_minor_summary_values_are_frozen():
    expected = {
        11: (7, 7, 630020914576076772),
        13: (94, 94, 1268914876423577257),
        14: (111, 111, 926543592233552319),
        15: (81, 81, 247473739368847072),
        16: (53, 53, 1822378321827871558),
        17: (28, 28, 1424445965610867005),
        18: (16, 16, 1996658450193783560),
        19: (7, 7, 1343131481176977680),
        20: (4, 4, 1674242889816756997),
        21: (1, 1, 1438514327499689729),
        22: (1, 1, 1381783072775710288),
    }

    observed = {
        item.chern_degree: (item.source_dimension, item.rank, item.expected_det_mod)
        for item in RANK5_PUBLIC_MINOR_SUMMARIES
    }
    assert observed == expected


def test_rank5_public_scalar_fixtures_have_top_degree_pairings():
    for fixture in RANK5_PUBLIC_SCALAR_FIXTURES:
        assert fixture.left.ordinary_degree == RANK5_SOURCE_DEGREE
        assert fixture.right.ordinary_degree == RANK5_TEST_DEGREE
        assert fixture.product.ordinary_degree == RANK5_TOP_DEGREE
        assert 0 <= fixture.expected_mod < fixture.prime


def test_rank5_fixture_lookup_by_name():
    fixture = scalar_fixture_by_name("f2_11__f2_13")

    assert fixture.left_name == "f2^11"
    assert fixture.right_name == "f2^13"


def test_rank5_c20_minor_fixture_shape_degrees_and_value():
    minor = RANK5_C20_MINOR_FIXTURE

    assert minor.shape == (4, 4)
    assert minor.chern_degree == 20
    assert tuple(item.ordinary_degree for item in minor.rows) == (22, 22, 22, 22)
    assert tuple(item.chern_degree for item in minor.rows) == (20, 20, 20, 20)
    assert tuple(item.ordinary_degree for item in minor.columns) == (26, 26, 26, 26)
    assert tuple(item.chern_degree for item in minor.columns) == (26, 24, 23, 22)
    assert minor.expected_det_mod == 1674242889816756997
    assert 0 <= minor.expected_det_mod < minor.prime


@pytest.mark.parametrize("minor", RANK5_SMALL_PUBLIC_MINOR_FIXTURES)
def test_rank5_small_public_minor_fixture_degrees(minor):
    assert minor.shape[0] == minor.shape[1]
    assert tuple(item.ordinary_degree for item in minor.rows) == (22,) * minor.shape[0]
    assert tuple(item.chern_degree for item in minor.rows) == (
        minor.chern_degree,
    ) * minor.shape[0]
    assert tuple(item.ordinary_degree for item in minor.columns) == (26,) * minor.shape[1]
    assert 0 <= minor.expected_det_mod < minor.prime


@pytest.mark.parametrize("fixture", RANK5_PUBLIC_SCALAR_FIXTURES)
def test_rank5_public_scalar_evaluator_regressions(fixture):
    assert slow_evaluator.pairing_mod_prime(
        RANK5_FORMULA,
        fixture.left,
        fixture.right,
        prime=fixture.prime,
    ) == fixture.expected_mod


def test_rank5_public_c20_minor_regression():
    minor = RANK5_C20_MINOR_FIXTURE
    matrix = slow_evaluator.pairing_matrix_mod_prime(
        RANK5_FORMULA,
        minor.rows,
        minor.columns,
        prime=minor.prime,
    )

    assert slow_evaluator.determinant_mod(matrix, minor.prime) == minor.expected_det_mod


@pytest.mark.parametrize("minor", RANK5_SMALL_PUBLIC_MINOR_FIXTURES)
def test_rank5_small_public_minor_regressions(minor):
    matrix = slow_evaluator.pairing_matrix_mod_prime(
        RANK5_FORMULA,
        minor.rows,
        minor.columns,
        prime=minor.prime,
    )

    assert slow_evaluator.determinant_mod(matrix, minor.prime) == minor.expected_det_mod
