# c18 Block Semantic Table Runbook

This is the Slurm-array path for gamma-sensitive c18 scouts.  It computes
unique semantic entries once for any currently supported row/test block, then
assembles the selected matrix rank by lookup.

Good first targets after the high-`f2` plateau:

- `ROW_KIND=gamma`, `COLUMN_KIND=f2-power`, `COLUMN_ORDER=f2-power-balanced`
- `ROW_KIND=gamma`, `COLUMN_KIND=one-f`
- `ROW_KIND=even`, `COLUMN_KIND=one-gamma`

Unsupported blocks such as `gamma x one-gamma` still raise by default.  Use
`--unsupported zero` only when you intentionally want unsupported entries
recorded as zero for plumbing checks.

## 1. Plan Semantic Keys

Example: gamma rows against low/balanced `f2` powers.

```bash
mkdir -p results/c18_block_semantic_gamma_f2_p101/chunks logs

PYTHONPATH=src python -m rank7_jk.c18_block_semantic_table plan-keys \
  --prime 101 \
  --method semantic-batched \
  --row-kind gamma \
  --column-kind f2-power \
  --column-order f2-power-balanced \
  --max-columns 256 \
  --chunk-size 100 \
  --output-dir results/c18_block_semantic_gamma_f2_p101/chunks \
  --output results/c18_block_semantic_gamma_f2_p101/manifest.json
```

For `gamma x one-f`, change `--column-kind one-f` and keep the default
`--column-order balanced`.  For `even x one-gamma`, use `--row-kind even` and
`--column-kind one-gamma`.

## 2. Local Smoke Test

```bash
PYTHONPATH=src RANK7_JK_DERIVATIVE_THREADS=1 \
python -m rank7_jk.c18_block_semantic_table run-key-manifest \
  results/c18_block_semantic_gamma_f2_p101/manifest.json \
  --task-id 0
```

## 3. Submit On Bouchet

Use the manifest `chunk_count` to choose the array range.

```bash
sbatch --array=0-356%24 \
  --cpus-per-task=1 \
  --mem=4G \
  --export=ALL,MANIFEST=$PWD/results/c18_block_semantic_gamma_f2_p101/manifest.json,REPO_ROOT=$PWD,PYTHON_BIN=$HOME/venvs/rank7/bin/python,PYTHON_MODULE=Python/3.12.3-GCCcore-13.3.0,DERIVATIVE_THREADS=1 \
  scripts/bouchet/submit_c18_block_semantic_keys.sbatch
```

The chunk worker skips existing complete outputs by default, so failed or
timed-out array tasks can be resubmitted safely.

## 4. Merge And Assemble

```bash
MANIFEST=$PWD/results/c18_block_semantic_gamma_f2_p101/manifest.json \
TABLE=$PWD/results/c18_block_semantic_gamma_f2_p101/value_table.json.gz \
OUTPUT=$PWD/results/c18_block_semantic_gamma_f2_p101/rank.json \
PYTHON_BIN=$HOME/venvs/rank7/bin/python \
PYTHON_MODULE=Python/3.12.3-GCCcore-13.3.0 \
scripts/bouchet/assemble_c18_block_semantic_table.sh
```

The rank artifact reports streamed rank, left nullity, selected independent
columns, per-column nonzero counts, and optional left-nullspace vectors.
