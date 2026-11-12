#!/bin/bash
# The whole chain on the weather family, chained by dependencies, from the repository root on a login node:
#   bash slurm/weather_chain.sh            # everything, from the law
#   bash slurm/weather_chain.sh vae        # from the command VAE (dataset already merged)
#   bash slurm/weather_chain.sh dagger     # from the DAgger collection (VAE and first distillation done)
# Results in $WORK/phase3_weather; the weather grids must be in data/weather/ (git has them).
set -euo pipefail
cd "$(dirname "$0")/.."
export STRIP=1 FAMILY=weather PHASE3_NAME=${PHASE3_NAME:-phase3_weather}
export PHYSICS_DIMS_OVERRIDE=16
source slurm/config.sh
cpu=(--account="$ACCOUNT" --partition="$CPU_PARTITION" --qos="$CPU_QOS" --cpus-per-task="$CPU_CORES")
serial=(--account="$ACCOUNT" --partition="$SERIAL_PARTITION" --qos="$SERIAL_QOS" --cpus-per-task="$SERIAL_CORES")
data="${WORK}/${PHASE3_NAME}"
logs="${data}/logs"
mkdir -p "$logs"
for day in $WEATHER_DAYS; do
    [ -f "data/weather/issoudun_$day.npz" ] || { echo "missing data/weather/issoudun_$day.npz" >&2; exit 1; }
done

submit() {  # sbatch wrapper: prints the job id; $1 = dependency (or "")
    local dep=$1; shift
    local id
    id=$(sbatch --parsable --chdir="$PWD" --export=ALL ${dep:+--dependency=afterok:$dep} "$@")
    echo "${id%%;*}"
}
from=${1:-all}
dep=""
if [ "$from" = "all" ]; then
    law=$(submit "" "${serial[@]}" --time="$TIME_LAW" --output="$logs/%x-%j.out" slurm/teacher_law.sbatch)
    shards=$(submit "$law" "${cpu[@]}" --time="$TIME_SHARD" --array=0-"$N_SHARDS" --output="$logs/%x-%A_%a.out" \
        slurm/teacher_shards.sbatch)
    dep=$(submit "$shards" "${serial[@]}" --time="$TIME_MERGE" --output="$logs/%x-%j.out" slurm/merge.sbatch)
    echo "law $law -> teacher $shards -> merge $dep"
fi
if [ "$from" = "all" ] || [ "$from" = "vae" ]; then
    vae=$(submit "$dep" "${cpu[@]}" --time="$TIME_PHYSICS" --output="$logs/%x-%j.out" slurm/physics.sbatch)
    dep=$(submit "$vae" "${cpu[@]}" --time="$TIME_DISTILL" --output="$logs/%x-%j.out" slurm/distill.sbatch)
    echo "command VAE $vae -> distill $dep"
fi
for step in collect distill fly; do
    dep=$(WEATHER_STEP=$step submit "$dep" "${cpu[@]}" --time="$TIME_WEATHER_STEP" --output="$logs/%x-%j.out" \
        slurm/weather_dagger.sbatch)
    echo "weather DAgger $step: $dep"
done
echo "Follow with:  squeue -u $USER      logs in $logs"
