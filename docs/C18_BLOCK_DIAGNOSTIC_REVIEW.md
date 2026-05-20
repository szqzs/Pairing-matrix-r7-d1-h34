# c18 Block Diagnostic Review

## Current Question

We want the even projection of the expected c18 relation, but the pairing
matrix may not reveal that projection if we only test the even source block.

Write the source slice schematically as

```text
H34_c18 = E + G
```

where `E` is the 126-dimensional `a/f` block and `G` is the 183-dimensional
`a/gamma` block.  If the true relation is

```text
R = R_even + R_gamma,
```

then a test class `t` gives

```text
<R_even, t> = - <R_gamma, t>.
```

Therefore an even-only matrix detects the even projection only when the chosen
test block annihilates the gamma source block, or when we include the gamma
source block and compute the block relation together.

## Implemented Diagnostic Blocks

The following rank-7/genus-2 c18-even test scaffolds now exist:

```text
all-a tests:        126 x 269
one-f tests:        126 x 1091
one-gamma tests:    126 x 2172
direct b-pair tests:126 x 28222
```

Observed sampled behavior so far:

```text
even x all-a:          sampled zero
even x one-f:          sampled zero
even x one-gamma:      sampled zero
even x direct b-pair:  sampled zero
gamma x all-a:         tiny sampled zero
```

The direct b-pair balanced probe

```text
12 defect-balanced rows x 12 mask-balanced columns
```

evaluated 144 real entries over `F_101`, all zero.

## What The Tests Do And Do Not Prove

Rank-5 regression tests now check the relevant kernel algebra:

- degree-two f-only delta kernels match the rank-5 reference;
- f+gamma delta kernels match the rank-5 reference;
- direct b-mask kernels match the rank-5 exterior-mask reference.

This validates the implemented formula pieces against the existing oracle, but
sampled zero entries do not prove global vanishing.

The main mathematical risk is block incompleteness.  Even if every sampled
even-row entry vanishes, this does not certify the even projection unless we
also understand the gamma block or prove the chosen test block annihilates it.

## Best Next Diagnostic

Compute block ranks rather than isolated samples.

Start with already-supported blocks:

```text
M_even,all-a
M_gamma,all-a
M_all,all-a
```

If `M_gamma,all-a` is nonzero, then the even-only all-a nullspace is not a
relation diagnostic.  If both even and gamma all-a blocks vanish, all-a is
blind and should not receive more cluster time.

Then test the degree-2 exterior blocks:

```text
M_even,one-gamma
M_gamma,one-f
M_even,direct-b-pair
```

The first and third have scaffolds.  The second is mathematically the same
total f+gamma shape but needs a gamma-source/one-f table wrapper.

Only after these block diagnostics show nonzero rank should we launch Bouchet
or rational reconstruction.

## Current Code Path

The direct b-mask adaptive probe is:

```bash
PYTHONPATH=src RANK7_JK_DERIVATIVE_THREADS=8 \
python -m rank7_jk.c18_b_mask_probe \
  --prime 101 \
  --method batched \
  --row-order defect-balanced \
  --column-order mask-balanced \
  --max-rows 12 \
  --max-columns 12 \
  --stop-on-nonzero \
  --output results/c18_b_mask_diagnostics/probe.json
```

Use this for diagnostic exploration, not as a complete matrix computation.
