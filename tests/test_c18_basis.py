from rank7_jk.c18_basis import (
    c18_even_source_rows,
    c18_gamma_source_rows,
    c18_source_rows,
    h62_all_a_test_columns,
    h62_one_f_test_columns,
    restricted_partitions,
)
from rank7_jk.config import RANK7_G2_D1


def test_restricted_partition_counts_for_fast_probe():
    parts = RANK7_G2_D1.class_ranks

    assert len(restricted_partitions(16, parts)) == 33
    assert len(restricted_partitions(14, parts)) == 23
    assert len(restricted_partitions(31, parts)) == 269


def test_c18_source_rows_split_into_even_and_gamma_defects():
    rows = c18_source_rows()
    even_rows = c18_even_source_rows()
    gamma_rows = c18_gamma_source_rows()

    assert len(rows) == 309
    assert len(even_rows) == 126
    assert len(gamma_rows) == 183
    assert rows == even_rows + gamma_rows

    assert even_rows[0].name == "a2^8 f2"
    assert even_rows[-1].name == "a5 a6 f7"
    assert gamma_rows[0].name == "a2^7 gamma22"
    assert gamma_rows[-1].name == "a4 gamma77"


def test_c18_source_rows_have_expected_degrees_and_defect_shape():
    for row in c18_source_rows():
        monomial = row.monomial
        assert monomial.ordinary_degree == 34
        assert monomial.chern_degree == 18
        assert sum(monomial.f_exp) + sum(monomial.gamma_exp) == 1
        if row.kind == "even":
            assert sum(monomial.f_exp) == 1
            assert not any(monomial.gamma_exp)
            assert row.defect.startswith("f")
        else:
            assert not any(monomial.f_exp)
            assert sum(monomial.gamma_exp) == 1
            assert row.defect.startswith("gamma")


def test_h62_all_a_test_columns_are_the_cheap_first_block():
    columns = h62_all_a_test_columns()

    assert len(columns) == 269
    assert columns[0].name == "a2^14 a3"
    assert columns[-1].name == "a6^4 a7"
    for column in columns:
        assert column.kind == "all_a"
        assert column.defect is None
        assert column.monomial.ordinary_degree == 62
        assert column.monomial.chern_degree == 31
        assert not any(column.monomial.f_exp)
        assert not any(column.monomial.gamma_exp)


def test_h62_one_f_test_columns_are_the_first_finish_candidate_block():
    columns = h62_one_f_test_columns()

    assert len(columns) == 1091
    assert columns[0].name == "a2^15 f2"
    assert columns[-1].name == "a6^3 a7 f7"
    for column in columns:
        assert column.kind == "one_f"
        assert column.defect and column.defect.startswith("f")
        assert column.monomial.ordinary_degree == 62
        assert column.monomial.chern_degree == 32
        assert sum(column.monomial.f_exp) == 1
        assert not any(column.monomial.gamma_exp)
