# Bouchet High-f2 Rank-Growth Runbook

This runbook is for the rank 7, genus 2, degree 1 c18 high-`f2` rank-growth
calculation.  It stores only mathematical source code and modular calculation
artifacts on Bouchet.  Do not store passwords, private keys, Yale confidential
data, personal data, or any other sensitive information in this repository or
in cluster result directories.

## Yale Policy Guardrails

- Use Bouchet only for low-risk research data for this project.
- Do not put SSH private keys, credentials, tokens, or personal/sensitive data
  on the cluster.
- Use `scratch` only for temporary files.  Scratch files older than 60 days may
  be purged, and artificial extension of scratch lifetime is not allowed without
  YCRC approval.
- Use compute nodes through Slurm for long jobs.  Login nodes are for editing,
  setup, short tests, and job submission.

Useful Yale documentation:

- Bouchet cluster: <https://docs.ycrc.yale.edu/clusters/bouchet/>
- Access: <https://docs.ycrc.yale.edu/clusters-at-yale/access/>
- SSH: <https://docs.ycrc.yale.edu/clusters-at-yale/access/ssh/>
- Transfer node: <https://docs.ycrc.yale.edu/data/transfer/>

## 1. Log In

From the Mac, while on Yale VPN:

```bash
ssh <netid>@bouchet.ycrc.yale.edu
```

If SSH is not set up yet, use Yale's SSH-key instructions.  Never share or
upload a private key.  Yale only needs the public key.

## 2. First Commands On Bouchet

```bash
groups
slurm_checkup.sh
mydirectories
module avail Python
```

`slurm_checkup.sh` shows the Slurm account/group setup.  `mydirectories` shows
the available home/project/scratch paths.

## 3. Put The Code On Bouchet

Recommended first transfer from the Mac:

```bash
rsync -av --delete \
  --exclude '.git' \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude 'logs/' \
  --exclude 'results/' \
  "$LOCAL_RANK7_REPO/" \
  <netid>@transfer-bouchet.ycrc.yale.edu:~/rank7/
```

The transfer node is the right place for moving files into Bouchet from the
Mac.  The excluded directories avoid uploading local logs/results and cache
files.

## 4. Create The Python Environment

On Bouchet:

```bash
cd ~/rank7
module avail Python
module load Python/3.12.3-GCCcore-13.3.0
python -m venv ~/venvs/rank7
source ~/venvs/rank7/bin/activate
pip install -U pip
pip install -e ".[speed,dev]"
```

If `Python/3.12.3-GCCcore-13.3.0` is not available, choose a nearby Python
3.10+ module shown by `module avail Python`, and pass the same module name as
`PYTHON_MODULE` when submitting Slurm jobs.

## 5. Smoke Test

Run a small test on the login node:

```bash
cd ~/rank7
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/rank7/bin/activate
pytest tests/test_c18_f2_rank_growth.py -q
```

This should take only a few seconds.  Do not run long rank-growth jobs on the
login node.

## 6. Submit The Cross-Prime Pilot

This is the first real Bouchet job:

```bash
cd ~/rank7
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/rank7/bin/activate

sbatch --time=04:00:00 \
  --cpus-per-task=8 \
  --mem=32G \
  --export=ALL,REPO_ROOT=$PWD,PYTHON_BIN=$VIRTUAL_ENV/bin/python,PYTHON_MODULE=Python/3.12.3-GCCcore-13.3.0,PRIME=1009,STOP_RANK=8,OUTPUT=$PWD/results/c18_f2_rank_growth/bouchet_p1009_rank8.json \
  scripts/bouchet/submit_c18_f2_rank_growth.sbatch
```

Monitor it:

```bash
squeue -u <netid>
tail -f logs/c18_f2_<jobid>.out
```

Inspect output after completion:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("results/c18_f2_rank_growth/bouchet_p1009_rank8.json")
data = json.loads(path.read_text())
print("rank", data["rank"], "processed", data["processed_columns"])
print("left_nullity", data["left_nullity"], "elapsed", data["elapsed_seconds"])
print("nonzero", data["nonzero_entries"], "/", data["attempted_entries"])
PY
```

## 7. Submit The Full p=101 Rank-308 Run

Only do this after the `p=1009` pilot works.

```bash
cd ~/rank7
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/rank7/bin/activate

sbatch --time=1-00:00:00 \
  --cpus-per-task=8 \
  --mem=32G \
  --export=ALL,REPO_ROOT=$PWD,PYTHON_BIN=$VIRTUAL_ENV/bin/python,PYTHON_MODULE=Python/3.12.3-GCCcore-13.3.0,PRIME=101,STOP_RANK=308,OUTPUT=$PWD/results/c18_f2_rank_growth/bouchet_p101_rank308.json \
  scripts/bouchet/submit_c18_f2_rank_growth.sbatch
```

The script automatically resumes from its checkpoint if the checkpoint exists.
If a job times out, resubmit the same command.

## 8. Bring Results Back To The Mac

From the Mac:

```bash
rsync -av \
  <netid>@transfer-bouchet.ycrc.yale.edu:~/rank7/results/c18_f2_rank_growth/ \
  "$LOCAL_RANK7_REPO/results/c18_f2_rank_growth/bouchet/"
```

Only result JSON/checkpoint/log files should come back.  Do not transfer any
cluster credentials or unrelated data.
