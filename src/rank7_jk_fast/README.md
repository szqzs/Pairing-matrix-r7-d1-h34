# rank7_jk_fast

Experimental c18 speed orchestration.  This folder does not edit or replace
the verified `rank7_jk` formula code; it wraps the existing evaluators with
run layouts that are easier to parallelize, resume, and abandon when rank
growth plateaus.

## What Changed

- Prefer semantic-key manifests over one long inline rank-growth process.
- Default row order is `sequential` for better defect/cache locality.
- Rank assembly can stop after `--max-dependent-columns` dependent columns.
- Worker commands set BLAS thread defaults to one to avoid oversubscription.
- A scout suite creates separate manifests for the likely useful blocks:
  `all x f2-power`, `gamma x f2-power`, `even x one-gamma`,
  `gamma x one-f`, and `even x b-pair`.

## Tiny Local Smoke

This uses the synthetic evaluator and two rows/columns, so it should stay
small:

```bash
rank7-c18-fast-semantic plan \
  --method synthetic \
  --row-kind all \
  --column-kind f2-power \
  --max-rows 2 \
  --max-columns 2 \
  --chunk-size 10 \
  --output-dir /tmp/r7fast/chunks \
  --output /tmp/r7fast/manifest.json

rank7-c18-fast-semantic run-chunk \
  /tmp/r7fast/manifest.json \
  --task-id 0 \
  --derivative-threads 1

rank7-c18-fast-semantic merge \
  /tmp/r7fast/manifest.json \
  --output /tmp/r7fast/value_table.json

rank7-c18-fast-semantic rank \
  /tmp/r7fast/value_table.json \
  --output /tmp/r7fast/rank.json \
  --max-dependent-columns 2
```

## Planning Real Scout Manifests

This only writes manifests; it does not evaluate entries:

```bash
rank7-c18-fast-semantic plan-suite \
  --prime 101 \
  --output-root results/fast_c18_scouts/p101
```

For one focused high-`f2` manifest:

```bash
rank7-c18-fast-semantic plan \
  --prime 101 \
  --row-kind all \
  --column-kind f2-power \
  --row-order sequential \
  --column-order f2-power-balanced \
  --max-columns 128 \
  --chunk-size 100 \
  --output-dir results/fast_c18_scouts/p101/all_f2/chunks \
  --output results/fast_c18_scouts/p101/all_f2/manifest.json
```

Evaluate chunks later with `run-chunk`, merge with `merge`, and inspect rank
with `rank`.  Use fresh output directories for different row/column orders.
