#!/bin/bash
# Phase 6, DAgger round: one step of the chain submitted by `submit_all.sh dagger` (DAGGER_STEP).
#   collect - the GA climbs on the 20 km tasks, every climb plan recorded   -> $DATA/dagger/collect/
#   distill - the generator retrained on the S1 GA solutions + the climbs  -> $DATA/dagger/distill/
#   fly     - the 20 km tasks with that generator, compared with the earlier pilots -> $DATA/dagger/crosscountry/
#SBATCH --job-name=gw-dagger
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

D="$DATA/dagger"
case "$DAGGER_STEP" in
collect)
    run python scenarios/phase5_crosscountry.py --vae "$DATA/physics/command_vae_d16.pt" --tasks "$XC_TASKS" \
        --search-budget "$XC_SEARCH_BUDGET" --pilots ga0.5 ga1 --record --out "$D/collect" --workers "$CORES" ;;
distill)
    run python scenarios/phase6_distill.py --data "$DATA/dataset.npz" --vae "$DATA/physics/command_vae_d16.pt" \
        --cache-from "$DATA/distill/ga_train.npz" --inflight "$D/collect/ga_inflight.npz" --out "$D/distill" \
        --workers "$CORES" ;;
fly)
    run python scenarios/phase5_crosscountry.py --vae "$DATA/physics/command_vae_d16.pt" --tasks "$XC_TASKS" \
        --search-budget "$XC_SEARCH_BUDGET" --pilots fast0.5 fast1 --flow "$D/distill/distilled_flow.pt" \
        --baseline "$DATA/crosscountry_fast/crosscountry.json" --out "$D/crosscountry" --workers "$CORES" ;;
*)
    echo "DAGGER_STEP must be collect, distill or fly" >&2; exit 1 ;;
esac
