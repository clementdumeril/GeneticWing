#!/bin/bash
# Phase 6: distill the latent GA into a fast generator (GA on the training situations, flow matching, test), one CPU node.
# Needs $DATA/dataset.npz and $DATA/physics/command_vae_d16.pt. Writes $DATA/distill/.
#SBATCH --job-name=gw-distill
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

run python scenarios/phase6_distill.py --data "$DATA/dataset.npz" --vae "$DATA/physics/command_vae_d16.pt" \
    --train "$DISTILL_TRAIN" --out "$DATA/distill" --workers "$CORES"
