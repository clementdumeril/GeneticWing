#!/bin/bash
# Plan with the point mass, fly with the 6-DOF rigid body (scenarios/sixdof_compare.py), on one CPU node.
# Needs the weather chain's artifacts: $DATA/theta.npy, $DATA/physics/command_vae_d16.pt and
# $DATA/dagger/distill/distilled_flow.pt (PHASE3_NAME=phase3_weather). Writes $DATA/sixdof/<run>/.
#   SIXDOF_PLANNERS  planners to fly (default: law teacher ga fast)
#   SIXDOF_W20       Dryden turbulence intensity, m/s (default 0: calm)
#   SIXDOF_INERTIA   inertia scale for the sensitivity run (default 1)
#SBATCH --job-name=gw-sixdof
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

planners=(${SIXDOF_PLANNERS:-law teacher ga fast})
w20=${SIXDOF_W20:-0}
inertia=${SIXDOF_INERTIA:-1}
name="w20_${w20}_inertia_${inertia}"
run python scenarios/sixdof_compare.py --theta "$DATA/theta.npy" --vae "$DATA/physics/command_vae_d16.pt" \
    --flow "$DATA/dagger/distill/distilled_flow.pt" --planners "${planners[@]}" --count "$SIXDOF_COUNT" \
    --w20 "$w20" --inertia-scale "$inertia" --search-budget "$XC_SEARCH_BUDGET" \
    --out "$DATA/sixdof/$name" --workers "$CORES"
