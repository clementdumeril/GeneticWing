#!/bin/bash
# Sensitivity of the strip-wing comparison to the SB-XC's estimated polar, on one CPU node.
# Needs $DATA/dataset.npz, $DATA/theta.npy and $DATA/physics/command_vae_d16.pt
# (the strip dataset and its command VAE). Writes $DATA/sensitivity/.
#SBATCH --job-name=gw-sensitivity
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

run python scenarios/phase3_realism.py --strip --radii 0 --airframes ${SENSITIVITY_AIRFRAMES[@]} \
    --data "$DATA/dataset.npz" --theta "$DATA/theta.npy" --vae "$DATA/physics/command_vae_d16.pt" \
    --out "$DATA/sensitivity" --workers "$CORES"
