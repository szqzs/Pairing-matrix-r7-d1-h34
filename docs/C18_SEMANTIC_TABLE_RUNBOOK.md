# c18 Semantic Table Runbook

This is the faster production path for the c18 even/all-a candidate relation.
It computes each unique semantic pairing value once, then assembles the matrix
by lookup.

## Current Dimensions

- c18 even rows: 126
- all-a H62 columns: 269
- raw matrix entries: 33,894
- unique semantic values: 6,086
- reuse factor: about 5.57x

## 1. Plan Semantic Keys

```bash
mkdir -p results/c18_semantic_p101/chunks logs

PYTHONPATH=src python -m rank7_jk.c18_semantic_table plan-keys \
  --prime 101 \
  --method semantic-batched \
  --row-kind even \
  --chunk-size 100 \
  --output-dir results/c18_semantic_p101/chunks \
  --output results/c18_semantic_p101/manifest.json
```

For the full even/all-a calculation this should create 61 chunks.

## 2. Local Smoke Test

```bash
PYTHONPATH=src RANK7_JK_DERIVATIVE_THREADS=7 \
python -m rank7_jk.c18_semantic_table run-key-manifest \
  results/c18_semantic_p101/manifest.json \
  --task-id 0
```

## 3. Submit On Bouchet

```bash
sbatch --array=0-60 \
  --export=ALL,MANIFEST=$PWD/results/c18_semantic_p101/manifest.json,REPO_ROOT=$PWD,DERIVATIVE_THREADS=7 \
  scripts/bouchet/submit_c18_semantic_keys.sbatch
```

The chunk worker skips an existing complete output by default, so failed or
timed-out array tasks can be resubmitted safely.

## 4. Merge And Assemble

```bash
MANIFEST=$PWD/results/c18_semantic_p101/manifest.json \
TABLE=$PWD/results/c18_semantic_p101/value_table.json.gz \
OUTPUT=$PWD/results/c18_semantic_p101/rank.json \
scripts/bouchet/assemble_c18_semantic_table.sh
```

The rank artifact reports the streamed rank, left nullity, selected independent
columns, and, by default, a left-nullspace basis.  If the expected candidate is
visible in the even projection, repeat over another prime such as 1009.
