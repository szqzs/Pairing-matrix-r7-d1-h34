# Rank 7 JK Pairing Project

This is a new codebase for the rank 7, genus 2, determinant degree 1
Jeffrey-Kirwan pairing calculation. It does not edit or import the old rank 5
frozen source. The first goal is a transparent mathematical reference layer
with structural checks before any optimization or cluster work.

Current target:

- rank `7`
- genus `2`
- determinant degree `1`
- source degree `34`
- test degree `62`
- expected candidate relation slice `c18`

Run the Step 1 checks:

```bash
PYTHONPATH=src python -m rank7_jk.checks
pytest
```

Gate B rank-5 regression is now implemented:

- `rank7_jk.invariants` parses and degree-checks monomials in `a_r`, `f_r`,
  and `gamma_rs`.
- `rank7_jk.rank5_regression` freezes the public rank-5 scalar and `c20`
  selected-minor fixtures.
- `rank7_jk.slow_evaluator.pairing_mod_prime` is a rank-5 genus-2
  semi-symbolic modular reference evaluator. It reproduces the frozen public
  scalar fixtures and the public `c19` through `c22` selected-minor
  determinants. Larger public determinant summaries are frozen as metadata for
  later optional slow checks.
- `rank7_jk.residue_oracle` separates literal SymPy Laurent checks for tiny
  ranks from the exact rational transition oracle used for larger spot checks.
- `rank7_jk.repro` writes a schema-validated Gate B artifact with source-tree
  hashes, old rank-5 certificate hashes, and environment metadata.

Write the Gate B artifact:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m rank7_jk.repro gate-b \
  --output artifacts/math_gates/gate_B_rank5_regression.json
```

Gate artifacts are generated outputs and are ignored by git. Gate B can use a
local read-only copy of the old rank-5 study repo through `RANK5_STUDY_REPO`;
when that repo is absent, committed provenance hashes are used.

Gate C rank-7 residue-transition smoke is now implemented:

- `rank7_jk.root_system` fixes the Type-A positive-root interval conventions
  with both zero-based implementation labels and one-based paper labels.
- `rank7_jk.sparse_poly` and `rank7_jk.mod_arith` provide small reusable
  finite-field kernels for the future fast evaluator.
- `rank7_jk.residue_transition` evaluates generic modular residue transitions
  and is checked against the rank-5 frozen transition path, exact tiny rank-7
  oracle cases, and two-prime rank-7 root-power-2 smoke fixtures.
- `rank7_jk.rank7_smoke` freezes the current rank-7 smoke cases.
- `rank7_jk.modular_formula` is the first rank-generic evaluator layer:
  modular `tau_r`, gradients, Hessians, B-map perturbations, c-tilde
  coefficients, and first hat-pair coefficients checked against
  `formula_ref`.

Write the Gate C artifact:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m rank7_jk.repro gate-c \
  --output artifacts/math_gates/gate_C_rank7_smoke.json
```
