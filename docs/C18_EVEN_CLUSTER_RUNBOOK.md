# c18 Even Cluster Runbook

This is the current production path for the c18 even/all-a probe.

## Goal

Compute all-a H62 column vectors against the c18 even source rows, merge the
chunks, and stop once the streamed rank/nullity gives the desired candidate
relation evidence.

Current dimensions:

- c18 rows: 309 total
- c18 even rows: 126
- all-a H62 columns: 269

## 1. Create A Manifest

From the repository root:

```bash
mkdir -p results/c18_even_p101/chunks logs

PYTHONPATH=src python -m rank7_jk.c18_even_worker plan \
  --prime 101 \
  --method semantic-batched \
  --row-kind even \
  --chunk-size 5 \
  --output-dir results/c18_even_p101/chunks \
  --output results/c18_even_p101/manifest.json
```

For 269 all-a columns and `--chunk-size 5`, this creates 54 chunks with task
ids `0-53`.

## 2. Local Smoke Test

Run one chunk locally before submitting the array:

```bash
PYTHONPATH=src RANK7_JK_DERIVATIVE_THREADS=7 \
python -m rank7_jk.c18_even_worker run-manifest \
  results/c18_even_p101/manifest.json \
  --task-id 0
```

Then merge partial outputs with missing chunks allowed:

```bash
PYTHONPATH=src python -m rank7_jk.c18_even_worker merge-manifest \
  results/c18_even_p101/manifest.json \
  --allow-missing \
  --output results/c18_even_p101/partial_merge.json
```

## 3. Submit A Slurm Array

On Bouchet, after cloning/updating the repository and activating the Python
environment:

```bash
sbatch --array=0-53 \
  --export=ALL,MANIFEST=$PWD/results/c18_even_p101/manifest.json,REPO_ROOT=$PWD,DERIVATIVE_THREADS=7 \
  scripts/bouchet/submit_c18_even_array.sbatch
```

Each task writes exactly one `.json.gz` chunk. Completed chunks are marked with
`"complete": true`; rerunning a task skips an existing complete chunk unless
`--recompute` is passed to `run-manifest`.

## 4. Merge Results

After all array tasks finish:

```bash
PYTHONPATH=src python -m rank7_jk.c18_even_worker merge-manifest \
  results/c18_even_p101/manifest.json \
  --output results/c18_even_p101/merged_rank.json
```

The merged artifact reports:

- rank
- left nullity
- selected independent column indices
- missing chunk outputs
- duplicate column indices
- git hash and dirty flag

## 5. First Mathematical Target

Start with:

- prime `101`
- row kind `even`
- method `semantic-batched`
- all 269 all-a columns

If the even projection is promising, repeat with another prime such as `1009`.

## 6. Repeat Over Primes

Make one result directory per prime:

```text
results/c18_even_p101/
results/c18_even_p1009/
```

Use the same manifest/chunk/merge flow for each prime.  Compare ranks,
left-nullities, and eventually candidate null vectors across primes.
