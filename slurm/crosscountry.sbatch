#!/bin/bash
# Phase 5: the cross-country benchmark (20 km tasks, classic circling against planned climbs), one CPU node.
# Needs $DATA/physics/command_vae_d16.pt. Writes $DATA/crosscountry/.
#SBATCH --job-name=gw-xc
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

flow=()
if [ -f "$DATA/distill/distilled_flow.pt" ]; then
    flow=(--flow "$DATA/distill/distilled_flow.pt")  # the fast pilots: the generator distilled from the GA
fi
run python scenarios/phase5_crosscountry.py --vae "$DATA/physics/command_vae_d16.pt" --tasks "$XC_TASKS" \
    --search-budget "$XC_SEARCH_BUDGET" --out "$DATA/${XC_OUT:-crosscountry}" --workers "$CORES" \
    ${flow[@]+"${flow[@]}"}
