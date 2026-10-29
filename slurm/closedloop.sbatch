#!/bin/bash
# Phase 4: the planners flown in closed loop (turbulence, wrong belief), on one CPU node.
# Needs $DATA/dataset.npz, $DATA/theta.npy and $DATA/physics/command_vae_d16.pt. Writes $DATA/closedloop/.
#SBATCH --job-name=gw-closedloop
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

run python scenarios/phase4_closed_loop.py --data "$DATA/dataset.npz" --theta "$DATA/theta.npy" \
    --vae "$DATA/physics/command_vae_d16.pt" --out "$DATA/closedloop" --workers "$CORES"
