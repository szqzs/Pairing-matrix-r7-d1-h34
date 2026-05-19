"""Exterior algebra and gamma conventions for JK odd variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .config import JKConfig

BLabel = Tuple[int, int]
MaskPoly = Dict[int, int]


@dataclass(frozen=True)
class ExteriorAlgebra:
    """Exterior algebra on the paper odd classes b_r^j."""

    config: JKConfig

    @property
    def b_labels(self) -> Tuple[BLabel, ...]:
        return self.config.b_labels

    @property
    def b_index(self) -> Dict[BLabel, int]:
        return {label: idx for idx, label in enumerate(self.b_labels)}

    def mask_for_b_label(self, label: BLabel) -> int:
        try:
            return 1 << self.b_index[label]
        except KeyError as exc:
            raise ValueError(f"invalid b label {label}") from exc

    def labels_from_mask(self, mask: int) -> Tuple[BLabel, ...]:
        return tuple(
            label
            for idx, label in enumerate(self.b_labels)
            if mask & (1 << idx)
        )

    def wedge_masks(self, left: int, right: int) -> Optional[Tuple[int, int]]:
        """Return (sign, mask) for left wedge right, or None if it is zero."""

        if left & right:
            return None
        inversions = 0
        active = left
        while active:
            bit = active & -active
            idx = bit.bit_length() - 1
            inversions += (right & ((1 << idx) - 1)).bit_count()
            active -= bit
        return (-1 if inversions % 2 else 1, left | right)

    def b_product_to_mask(self, labels: Sequence[BLabel]) -> Optional[Tuple[int, int]]:
        mask = 0
        sign = 1
        for label in labels:
            step = self.wedge_masks(mask, self.mask_for_b_label(label))
            if step is None:
                return None
            step_sign, mask = step
            sign *= step_sign
        return sign, mask

    def multiply(self, left: MaskPoly, right: MaskPoly) -> MaskPoly:
        out: MaskPoly = {}
        for left_mask, left_coeff in left.items():
            for right_mask, right_coeff in right.items():
                wedge = self.wedge_masks(left_mask, right_mask)
                if wedge is None:
                    continue
                sign, mask = wedge
                value = out.get(mask, 0) + sign * left_coeff * right_coeff
                if value:
                    out[mask] = value
                else:
                    out.pop(mask, None)
        return out

    def gamma_terms(self, r: int, s: int) -> Tuple[Tuple[int, Tuple[BLabel, ...]], ...]:
        """Paper-level symplectic abbreviation gamma_rs in b variables."""

        if r < 2 or r > self.config.rank or s < 2 or s > self.config.rank:
            raise ValueError(f"invalid gamma label {(r, s)} for rank {self.config.rank}")
        terms = []
        for i in range(1, self.config.genus + 1):
            terms.append((1, ((r, i), (s, i + self.config.genus))))
            terms.append((-1, ((r, i + self.config.genus), (s, i))))
        return tuple(terms)

    def gamma_as_mask_poly(self, r: int, s: int) -> MaskPoly:
        out: MaskPoly = {}
        for coeff, labels in self.gamma_terms(r, s):
            target = self.b_product_to_mask(labels)
            if target is None:
                continue
            sign, mask = target
            value = out.get(mask, 0) + coeff * sign
            if value:
                out[mask] = value
            else:
                out.pop(mask, None)
        return out

    def gamma_product_to_mask_poly(self, gamma_exp: Sequence[int]) -> MaskPoly:
        if len(gamma_exp) != len(self.config.gamma_labels):
            raise ValueError(
                f"expected {len(self.config.gamma_labels)} gamma exponents, "
                f"got {len(gamma_exp)}"
            )
        out: MaskPoly = {0: 1}
        for idx, exp in enumerate(gamma_exp):
            if not exp:
                continue
            r, s = self.config.gamma_labels[idx]
            factor = self.gamma_as_mask_poly(r, s)
            for _ in range(int(exp)):
                out = self.multiply(out, factor)
                if not out:
                    return {}
        return out

    def gamma_product_to_b_terms(
        self,
        gamma_exp: Sequence[int],
    ) -> Tuple[Tuple[int, Tuple[BLabel, ...]], ...]:
        poly = self.gamma_product_to_mask_poly(gamma_exp)
        return tuple(
            (coeff, self.labels_from_mask(mask))
            for mask, coeff in sorted(poly.items())
            if coeff
        )


def gamma_exponent(config: JKConfig, label: Tuple[int, int], power: int = 1) -> Tuple[int, ...]:
    values = [0 for _ in config.gamma_labels]
    try:
        idx = config.gamma_labels.index(label if label[0] <= label[1] else (label[1], label[0]))
    except ValueError as exc:
        raise ValueError(f"invalid gamma label {label}") from exc
    values[idx] = int(power)
    return tuple(values)
