#!/bin/bash
# Tune the reactive law once; every shard reuses it (writes $DATA/theta.npy).
# Always on the point-mass wing, also for STRIP=1: the law only seeds the NLP.
#SBATCH --job-name=gw-law
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

run python scenarios/phase3_dataset.py ${strip_flag[@]+"${strip_flag[@]}"} --tune-law-only --out "$DATA"
