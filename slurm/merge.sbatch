#!/bin/bash
# Merge the shards into $DATA/dataset.npz.
#SBATCH --job-name=gw-merge
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

run python scenarios/phase3_merge.py "$DATA/shards" --out "$DATA/dataset.npz"
