#!/bin/bash
# Latent GA with the physics decoder (command VAE + simulator), on one whole CPU node.
# Writes $DATA/physics/ (summary, weights).
#SBATCH --job-name=gw-physics
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

margins=()
if [ "${PHYSICS_MARGINS:-0}" = "1" ]; then
    margins=(--margins)  # commands bounded by the teacher's margins: same feasible set as the teacher
fi
run python scenarios/phase3_physics.py --data "$DATA/dataset.npz" --out "$DATA/physics" \
    --latent-dims ${PHYSICS_DIMS[@]} --steps "$LATENT_STEPS" ${margins[@]+"${margins[@]}"} \
    --workers "$CORES" --threads "$CORES" --device cpu
