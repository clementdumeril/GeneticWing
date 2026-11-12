#!/bin/bash
# The weather chain's DAgger round, one step per job (WEATHER_STEP), in the thermals of the real days:
#   collect - blind GA climbs on the training tasks, every climb plan recorded -> $DATA/dagger/collect/<day>/
#   distill - the generator retrained on the family's GA solutions + those climbs -> $DATA/dagger/distill/
#   fly     - the held-out tasks: blind generator, generator with the map, and the rules -> $DATA/dagger/test/<day>/
#SBATCH --job-name=gw-wdagger
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

D="$DATA/dagger"
case "$WEATHER_STEP" in
collect)
    for day in $WEATHER_DAYS; do
        run python scenarios/phase5_crosscountry.py --vae "$DATA/physics/command_vae_d16.pt" \
            --weather "data/weather/issoudun_$day.npz" --first "$WEATHER_COLLECT_FIRST" --tasks "$WEATHER_COLLECT_TASKS" \
            --search-budget "$XC_SEARCH_BUDGET" --pilots bga0.5 bga1 --record --out "$D/collect/$day" --workers "$CORES"
    done ;;
distill)
    inflight=()
    for day in $WEATHER_DAYS; do inflight+=("$D/collect/$day/ga_inflight.npz"); done
    run python scenarios/phase6_distill.py --data "$DATA/dataset.npz" --vae "$DATA/physics/command_vae_d16.pt" \
        --cache-from "$DATA/distill/ga_train.npz" --inflight "${inflight[@]}" --out "$D/distill" --workers "$CORES" ;;
fly)
    for day in $WEATHER_DAYS; do
        run python scenarios/phase5_crosscountry.py --vae "$DATA/physics/command_vae_d16.pt" \
            --weather "data/weather/issoudun_$day.npz" --tasks "$WEATHER_TEST_TASKS" \
            --search-budget "$XC_SEARCH_BUDGET" --pilots motor mc0.5 mc1 bmc0.5 bmc1 fast0.5 fast1 bfast0.5 bfast1 \
            --flow "$D/distill/distilled_flow.pt" --out "$D/test/$day" --workers "$CORES"
    done ;;
*)
    echo "WEATHER_STEP must be collect, distill or fly" >&2; exit 1 ;;
esac
