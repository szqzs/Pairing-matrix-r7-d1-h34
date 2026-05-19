"""Invariant monomial bookkeeping for JK pairing inputs.

This layer models the paper-level Sp-invariant monomials in the even classes
``a_r``, ``f_r`` and the abbreviations ``gamma_rs``.  It intentionally does no
JK evaluation; it is the degree/parser contract that later evaluators consume.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence, Tuple

from .config import FormulaConfig


class InvariantParseError(ValueError):
    """Raised when a monomial string is not a valid invariant expression."""


def class_ranks(rank: int) -> Tuple[int, ...]:
    if rank < 2:
        raise ValueError("rank must be at least 2")
    return tuple(range(2, rank + 1))


def gamma_labels(rank: int) -> Tuple[Tuple[int, int], ...]:
    if rank < 2:
        raise ValueError("rank must be at least 2")
    return tuple((r, s) for r in range(2, rank + 1) for s in range(r, rank + 1))


def _check_exp_tuple(
    label: str,
    values: Sequence[int],
    expected_len: int,
) -> Tuple[int, ...]:
    if len(values) != expected_len:
        raise ValueError(f"{label} exponent length {len(values)} != {expected_len}")
    out = tuple(int(item) for item in values)
    if any(item < 0 for item in out):
        raise ValueError(f"{label} exponents must be nonnegative")
    return out


def _format_factor(name: str, power: int) -> str:
    return name if power == 1 else f"{name}^{power}"


@dataclass(frozen=True)
class InvariantMonomial:
    """Exponents for a product of ``a_r``, ``f_r`` and ``gamma_rs`` classes."""

    rank: int
    a_exp: Tuple[int, ...]
    f_exp: Tuple[int, ...]
    gamma_exp: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.rank < 2:
            raise ValueError("rank must be at least 2")
        object.__setattr__(
            self,
            "a_exp",
            _check_exp_tuple("a", self.a_exp, self.rank - 1),
        )
        object.__setattr__(
            self,
            "f_exp",
            _check_exp_tuple("f", self.f_exp, self.rank - 1),
        )
        object.__setattr__(
            self,
            "gamma_exp",
            _check_exp_tuple("gamma", self.gamma_exp, len(gamma_labels(self.rank))),
        )

    @classmethod
    def identity(cls, config: FormulaConfig) -> "InvariantMonomial":
        return cls(
            rank=config.rank,
            a_exp=tuple(0 for _ in config.class_ranks),
            f_exp=tuple(0 for _ in config.class_ranks),
            gamma_exp=tuple(0 for _ in config.gamma_labels),
        )

    @classmethod
    def from_exponents(
        cls,
        config: FormulaConfig,
        *,
        a_exp: Sequence[int] | None = None,
        f_exp: Sequence[int] | None = None,
        gamma_exp: Sequence[int] | None = None,
    ) -> "InvariantMonomial":
        default_a = tuple(0 for _ in config.class_ranks)
        default_f = tuple(0 for _ in config.class_ranks)
        default_gamma = tuple(0 for _ in config.gamma_labels)
        return cls(
            rank=config.rank,
            a_exp=tuple(a_exp if a_exp is not None else default_a),
            f_exp=tuple(f_exp if f_exp is not None else default_f),
            gamma_exp=tuple(gamma_exp if gamma_exp is not None else default_gamma),
        )

    @classmethod
    def from_string(cls, config: FormulaConfig, text: str) -> "InvariantMonomial":
        a_exp = [0 for _ in config.class_ranks]
        f_exp = [0 for _ in config.class_ranks]
        gamma_exp = [0 for _ in config.gamma_labels]
        gamma_index = {label: idx for idx, label in enumerate(config.gamma_labels)}

        stripped = text.strip()
        if stripped in {"", "1"}:
            return cls.from_exponents(config)

        for token in re.split(r"[\s*]+", stripped):
            if not token:
                continue
            base, power = _split_power(token)
            if power == 0:
                continue

            kind, label = _parse_base(config, base)
            if kind == "a":
                a_exp[int(label) - 2] += power
            elif kind == "f":
                f_exp[int(label) - 2] += power
            else:
                gamma_exp[gamma_index[label]] += power

        return cls.from_exponents(config, a_exp=a_exp, f_exp=f_exp, gamma_exp=gamma_exp)

    @property
    def is_identity(self) -> bool:
        return not any(self.a_exp) and not any(self.f_exp) and not any(self.gamma_exp)

    @property
    def ordinary_degree(self) -> int:
        total = 0
        for power, r in zip(self.a_exp, class_ranks(self.rank)):
            total += int(power) * 2 * r
        for power, r in zip(self.f_exp, class_ranks(self.rank)):
            total += int(power) * (2 * r - 2)
        for power, (r, s) in zip(self.gamma_exp, gamma_labels(self.rank)):
            total += int(power) * (2 * r + 2 * s - 2)
        return total

    @property
    def chern_degree(self) -> int:
        total = 0
        for power, r in zip(self.a_exp, class_ranks(self.rank)):
            total += int(power) * r
        for power, r in zip(self.f_exp, class_ranks(self.rank)):
            total += int(power) * r
        for power, (r, s) in zip(self.gamma_exp, gamma_labels(self.rank)):
            total += int(power) * (r + s)
        return total

    def multiply(self, other: "InvariantMonomial") -> "InvariantMonomial":
        if self.rank != other.rank:
            raise ValueError(f"cannot multiply rank {self.rank} by rank {other.rank}")
        return InvariantMonomial(
            rank=self.rank,
            a_exp=tuple(left + right for left, right in zip(self.a_exp, other.a_exp)),
            f_exp=tuple(left + right for left, right in zip(self.f_exp, other.f_exp)),
            gamma_exp=tuple(
                left + right for left, right in zip(self.gamma_exp, other.gamma_exp)
            ),
        )

    def __mul__(self, other: "InvariantMonomial") -> "InvariantMonomial":
        return self.multiply(other)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "a_exp": list(self.a_exp),
            "f_exp": list(self.f_exp),
            "gamma_exp": list(self.gamma_exp),
            "name": str(self),
            "ordinary_degree": self.ordinary_degree,
            "chern_degree": self.chern_degree,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InvariantMonomial":
        return cls(
            rank=int(payload["rank"]),
            a_exp=tuple(int(item) for item in payload["a_exp"]),
            f_exp=tuple(int(item) for item in payload["f_exp"]),
            gamma_exp=tuple(int(item) for item in payload["gamma_exp"]),
        )

    def sort_key(self) -> Tuple[int, int, str]:
        return (self.ordinary_degree, self.chern_degree, str(self))

    def __str__(self) -> str:
        factors = []
        for power, r in zip(self.a_exp, class_ranks(self.rank)):
            if power:
                factors.append(_format_factor(f"a{r}", power))
        for power, r in zip(self.f_exp, class_ranks(self.rank)):
            if power:
                factors.append(_format_factor(f"f{r}", power))
        for power, (r, s) in zip(self.gamma_exp, gamma_labels(self.rank)):
            if power:
                factors.append(_format_factor(f"gamma{r}{s}", power))
        return " ".join(factors) if factors else "1"


def _split_power(token: str) -> Tuple[str, int]:
    pieces = token.split("^")
    if len(pieces) > 2 or not pieces[0]:
        raise InvariantParseError(f"bad invariant token {token!r}")
    if len(pieces) == 1:
        return pieces[0], 1
    try:
        power = int(pieces[1])
    except ValueError as exc:
        raise InvariantParseError(f"bad exponent in token {token!r}") from exc
    if power < 0:
        raise InvariantParseError(f"negative exponent in token {token!r}")
    return pieces[0], power


def _parse_base(config: FormulaConfig, base: str) -> Tuple[str, int | Tuple[int, int]]:
    even_match = re.fullmatch(r"([af])(\d+)", base)
    if even_match:
        kind, rank_text = even_match.groups()
        r = int(rank_text)
        if r not in config.class_ranks:
            raise InvariantParseError(f"{base!r} is outside rank {config.rank}")
        return kind, r

    if not base.startswith("gamma"):
        raise InvariantParseError(f"unknown invariant factor {base!r}")

    r, s = _parse_gamma_tail(base[len("gamma") :])
    if r > s:
        r, s = s, r
    if (r, s) not in config.gamma_labels:
        raise InvariantParseError(f"{base!r} is outside rank {config.rank}")
    return "gamma", (r, s)


def _parse_gamma_tail(tail: str) -> Tuple[int, int]:
    cleaned = tail.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]

    if "," in cleaned:
        pieces = cleaned.split(",")
    elif "_" in cleaned:
        pieces = cleaned.split("_")
    elif cleaned.isdigit() and len(cleaned) == 2:
        pieces = [cleaned[0], cleaned[1]]
    else:
        raise InvariantParseError(
            "gamma labels need gamma22, gamma2_3, or gamma(2,3) syntax"
        )

    if len(pieces) != 2:
        raise InvariantParseError(f"bad gamma label {tail!r}")
    try:
        return int(pieces[0]), int(pieces[1])
    except ValueError as exc:
        raise InvariantParseError(f"bad gamma label {tail!r}") from exc
