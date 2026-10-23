#!/bin/bash
# Genetic search in the latent space of a conditional VAE, on one whole CPU node:
# VAEs trained with all cores, then latent searches spread over the cores.
# Writes $DATA/latent/ (summary, weights).
#SBATCH --job-name=gw-latent
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

python scenarios/phase3_latent.py --data "$DATA/dataset.npz" --out "$DATA/latent" \
    --latent-dims ${LATENT_DIMS[@]} --rounds "$LATENT_ROUNDS" --loop-solutions "$LOOP_SOLUTIONS" \
    --steps "$LATENT_STEPS" --workers "$(nproc)" --threads "$(nproc)" --device cpu
