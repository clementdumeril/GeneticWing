#!/bin/bash
# Tune the reactive law once; every shard reuses it (writes $DATA/theta.npy).
#SBATCH --job-name=gw-law
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

run python scenarios/phase3_dataset.py --tune-law-only --out "$DATA"
