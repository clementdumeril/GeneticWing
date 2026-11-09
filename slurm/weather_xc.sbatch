#!/bin/bash
# Cross-country tasks in the thermals of real days (geneticwing.weather): routes over Issoudun on three days
# of summer 2025, thermals from the archived ICON-D2 forecast with Allen's profile, born and dying in flight.
#   bash slurm/weather_xc.sh          (submits this script with the settings of slurm/config.sh)
#SBATCH --job-name=gw-weather
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source slurm/common.sh

D="$DATA/dagger"
for day in ${WEATHER_DAYS:-2025-07-12 2025-08-02 2025-06-22}; do
    run python scenarios/phase5_crosscountry.py --vae "$DATA/physics/command_vae_d16.pt" \
        --weather "data/weather/issoudun_$day.npz" --tasks "${WEATHER_TASKS:-32}" \
        --search-budget "$XC_SEARCH_BUDGET" --pilots motor mc0.5 mc1 ga0.5 ga1 fast0.5 fast1 \
        --flow "$D/distill/distilled_flow.pt" --out "$D/weather/$day" --workers "$CORES"
done
