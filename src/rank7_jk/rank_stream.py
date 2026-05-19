"""Modular rank and left-nullspace utilities for streamed pairing matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Tuple

from .mod_arith import mod_inv, require_prime


def matrix_rank_mod(matrix: Sequence[Sequence[int]], prime: int) -> int:
    _rank, _pivots, _rref = row_reduce_mod(matrix, prime)
    return _rank


def row_reduce_mod(
    matrix: Sequence[Sequence[int]],
    prime: int,
) -> Tuple[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]]:
    p = require_prime(prime)
    mat = [[int(value) % p for value in row] for row in matrix]
    if not mat:
        return 0, (), ()
    width = len(mat[0])
    if any(len(row) != width for row in mat):
        raise ValueError("matrix rows must all have the same length")

    rank = 0
    pivots = []
    for col in range(width):
        pivot = None
        for row in range(rank, len(mat)):
            if mat[row][col] % p:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
        inv = mod_inv(mat[rank][col], p)
        mat[rank] = [value * inv % p for value in mat[rank]]
        for row in range(len(mat)):
            if row == rank:
                continue
            factor = mat[row][col] % p
            if factor:
                mat[row] = [
                    (mat[row][idx] - factor * mat[rank][idx]) % p
                    for idx in range(width)
                ]
        pivots.append(col)
        rank += 1
        if rank == len(mat):
            break
    return rank, tuple(pivots), tuple(tuple(row) for row in mat)


def right_nullspace_mod(
    matrix: Sequence[Sequence[int]],
    prime: int,
) -> Tuple[Tuple[int, ...], ...]:
    p = require_prime(prime)
    if not matrix:
        return ()
    width = len(matrix[0])
    rank, pivots, rref = row_reduce_mod(matrix, p)
    pivot_set = set(pivots)
    free_columns = [idx for idx in range(width) if idx not in pivot_set]
    basis = []
    for free in free_columns:
        vector = [0 for _ in range(width)]
        vector[free] = 1
        for pivot_row, pivot_col in enumerate(pivots):
            vector[pivot_col] = (-rref[pivot_row][free]) % p
        basis.append(tuple(vector))
    if rank == width:
        return ()
    return tuple(basis)


def left_nullspace_mod(
    matrix: Sequence[Sequence[int]],
    prime: int,
) -> Tuple[Tuple[int, ...], ...]:
    return right_nullspace_mod(transpose(matrix), prime)


def transpose(matrix: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    if not matrix:
        return ()
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must all have the same length")
    return tuple(tuple(row[idx] for row in matrix) for idx in range(width))


def mat_vec_mul_mod(
    matrix: Sequence[Sequence[int]],
    vector: Sequence[int],
    prime: int,
) -> Tuple[int, ...]:
    p = require_prime(prime)
    out = []
    for row in matrix:
        if len(row) != len(vector):
            raise ValueError("matrix width must match vector length")
        out.append(sum(int(a) * int(b) for a, b in zip(row, vector)) % p)
    return tuple(out)


def vec_mat_mul_mod(
    vector: Sequence[int],
    matrix: Sequence[Sequence[int]],
    prime: int,
) -> Tuple[int, ...]:
    return mat_vec_mul_mod(transpose(matrix), vector, prime)


@dataclass
class ColumnRankTracker:
    """Incrementally track rank of a matrix whose columns are streamed."""

    row_count: int
    prime: int
    pivot_rows: list[int] = field(default_factory=list)
    basis_columns: list[list[int]] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)
    processed_columns: int = 0

    def __post_init__(self) -> None:
        if self.row_count < 0:
            raise ValueError("row_count must be nonnegative")
        self.prime = require_prime(self.prime)

    @property
    def rank(self) -> int:
        return len(self.pivot_rows)

    @property
    def nullity_left(self) -> int:
        return self.row_count - self.rank

    def add_column(self, column: Sequence[int], *, index: int | None = None) -> bool:
        if len(column) != self.row_count:
            raise ValueError("column length does not match row_count")
        p = self.prime
        vec = [int(value) % p for value in column]
        for pivot_row, basis in zip(self.pivot_rows, self.basis_columns):
            factor = vec[pivot_row] % p
            if factor:
                vec = [(vec[idx] - factor * basis[idx]) % p for idx in range(self.row_count)]
        pivot = next((idx for idx, value in enumerate(vec) if value % p), None)
        self.processed_columns += 1
        if pivot is None:
            return False
        inv = mod_inv(vec[pivot], p)
        vec = [value * inv % p for value in vec]
        for old_pivot, basis in zip(self.pivot_rows, self.basis_columns):
            factor = basis[pivot] % p
            if factor:
                for idx in range(self.row_count):
                    basis[idx] = (basis[idx] - factor * vec[idx]) % p
                basis[old_pivot] = 1
        self.pivot_rows.append(pivot)
        self.basis_columns.append(vec)
        self.selected_indices.append(self.processed_columns - 1 if index is None else int(index))
        order = sorted(range(self.rank), key=lambda idx: self.pivot_rows[idx])
        self.pivot_rows = [self.pivot_rows[idx] for idx in order]
        self.basis_columns = [self.basis_columns[idx] for idx in order]
        self.selected_indices = [self.selected_indices[idx] for idx in order]
        return True

    def add_columns(
        self,
        columns: Iterable[Sequence[int]],
        *,
        start_index: int = 0,
        stop_rank: int | None = None,
    ) -> int:
        added = 0
        for offset, column in enumerate(columns):
            if stop_rank is not None and self.rank >= stop_rank:
                break
            if self.add_column(column, index=start_index + offset):
                added += 1
        return added
