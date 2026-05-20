# c18 Fast Scout Runbook

This is the lightweight orchestration layer for short c18 scout runs.  It does
not replace the verified `rank7_jk` formula code; it calls those evaluators
through `rank7_jk_fast.semantic_pipeline` and keeps the run layout easy to
parallelize, resume, and abandon when a block plateaus.

Do not store sensitive data on Bouchet.  These commands write only code logs
and mathematical JSON artifacts under `results/`.

## 1. Plan The Scout Manifests

```bash
cd ~/rank7
git pull

OUTPUT_ROOT=$PWD/results/fast_c18_scouts/p101 \
PYTHON_BIN=$HOME/venvs/rank7/bin/python \
PYTHON_MODULE=Python/3.12.3-GCCcore-13.3.0 \
scripts/bouchet/plan_c18_fast_scout_suite.sh
```

The default suite currently plans:

- `all_f2_balanced_128`
- `gamma_f2_balanced_256`
- `even_one_gamma_128`
- `gamma_one_f_128`
- `even_b_pair_64`

Given the first `gamma x one-f16` scout had rank `0`, do not launch
`gamma_one_f_128` as a default production scout.  Use it only if we
deliberately decide to test a different one-f ordering/window.

## 2. Submit A Small Scout Array

Use the manifest `chunk_count` as the array size.  This snippet submits one
manifest with a conservative concurrency cap.

```bash
MANIFEST=$PWD/results/fast_c18_scouts/p101/even_b_pair_64/manifest.json
CHUNKS=$($HOME/venvs/rank7/bin/python - <<PY
import json
from pathlib import Path
print(json.loads(Path("$MANIFEST").read_text())["chunk_count"])
PY
)

sbatch --array=0-$((CHUNKS - 1))%16 \
  --time=06:00:00 \
  --cpus-per-task=1 \
  --mem=4G \
  --export=ALL,MANIFEST=$MANIFEST,REPO_ROOT=$PWD,PYTHON_BIN=$HOME/venvs/rank7/bin/python,PYTHON_MODULE=Python/3.12.3-GCCcore-13.3.0,DERIVATIVE_THREADS=1 \
  scripts/bouchet/submit_c18_fast_scout.sbatch
```

Recommended first scouts:

- `all_f2_balanced_128`
- `even_one_gamma_128`
- `even_b_pair_64`

The worker skips existing complete chunk files by default, so rerunning failed
array tasks is safe.

## 3. Merge And Assemble

For a final scout artifact, compute the exact rank over all selected columns:

```bash
MANIFEST=$PWD/results/fast_c18_scouts/p101/even_b_pair_64/manifest.json \
PYTHON_BIN=$HOME/venvs/rank7/bin/python \
PYTHON_MODULE=Python/3.12.3-GCCcore-13.3.0 \
NO_PLATEAU_STOP=1 \
scripts/bouchet/assemble_c18_fast_scout.sh
```

For quick local or partial inspection, omit `NO_PLATEAU_STOP=1`; the rank pass
then stops after `MAX_DEPENDENT_COLUMNS=32` dependent columns by default.

## 4. Read The Scoreboard

The rank JSON reports:

- `rank`
- `left_nullity`
- `processed_columns`
- `stop_reason`
- selected independent columns
- per-column nonzero counts

The next scale-up should be chosen from blocks that add new rank quickly.  A
rank-zero or plateauing block should not be expanded without a new mathematical
reason.
