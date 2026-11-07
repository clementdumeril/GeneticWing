#!/bin/bash
# Held-out check of the DAgger generator: the 20 km tasks 64..127, which no training step has seen
# (the DAgger round recorded its states on tasks 0..63). Every pilot flies the same new tasks.
#   bash slurm/heldout_xc.sh          (submits this script with the settings of slurm/config.sh)
#SBATCH --job-name=gw-heldout
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

D="$DATA/dagger"
run python scenarios/phase5_crosscountry.py --vae "$DATA/physics/command_vae_d16.pt" --tasks "$XC_TASKS" --first 64 \
    --search-budget "$XC_SEARCH_BUDGET" --pilots motor mc0.5 mc1 ga0.5 ga1 fast0.5 fast1 \
    --flow "$D/distill/distilled_flow.pt" --out "$D/crosscountry_heldout" --workers "$CORES"
