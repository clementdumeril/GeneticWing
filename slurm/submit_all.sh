#!/bin/bash
# Submit the whole phase 3 pipeline on Savio, chained by dependencies:
#   law (1 node) -> teacher shards (N_SHARDS + 1 nodes) -> merge -> students (1 GPU per size)
# From the repository root on a login node:
#   bash slurm/submit_all.sh            # everything
#   bash slurm/submit_all.sh train      # only the students (dataset already merged)
#   bash slurm/submit_all.sh latent     # only the latent GA (dataset already merged)
#   bash slurm/submit_all.sh physics    # only the latent GA with the physics decoder
#   bash slurm/submit_all.sh strip      # strip-theory wing: law -> shards -> merge -> physics,
#                                       # in $WORK/phase3_strip
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-all}" = "strip" ]; then
    export STRIP=1 PHASE3_NAME=phase3_strip
fi
source slurm/config.sh

if [[ "$ACCOUNT" == *CHANGEME* ]]; then
    echo "Edit slurm/config.sh first (ACCOUNT, partitions, QoS): see bash slurm/check_savio.sh" >&2
    exit 1
fi

cpu=(--account="$ACCOUNT" --partition="$CPU_PARTITION" --qos="$CPU_QOS" --cpus-per-task="$CPU_CORES")
serial=(--account="$ACCOUNT" --partition="$SERIAL_PARTITION" --qos="$SERIAL_QOS" --cpus-per-task="$SERIAL_CORES")
gpu=(--account="$ACCOUNT" --partition="$GPU_PARTITION" --qos="$GPU_QOS" --gres="$GPU_GRES" --cpus-per-task="$GPU_CPUS")
if [ "$TRAIN_ON" = "cpu" ]; then
    train_res=("${cpu[@]}")
else
    train_res=("${gpu[@]}")
fi
data="${WORK}/${PHASE3_NAME}"
logs="${data}/logs"
mkdir -p "$logs"

submit() {  # sbatch wrapper that prints and returns the job id
    local id
    id=$(sbatch --parsable --chdir="$PWD" "$@")
    id=${id%%;*}  # drop a cluster name, if any
    echo "$id"
}

n_sizes=${#TRAIN_SIZES[@]}
if [ "${1:-all}" = "physics" ]; then
    physics=$(submit "${cpu[@]}" --time="$TIME_PHYSICS" --output="$logs/%x-%j.out" slurm/physics.sbatch)
    echo "physics decoder: job $physics (results in ${data}/physics)"
    exit 0
fi
if [ "${1:-all}" = "latent" ]; then
    latent=$(submit "${cpu[@]}" --time="$TIME_LATENT" --output="$logs/%x-%j.out" slurm/latent.sbatch)
    echo "latent GA: job $latent (results in ${data}/latent)"
    exit 0
fi
if [ "${1:-all}" = "train" ]; then
    train=$(submit "${train_res[@]}" --time="$TIME_TRAIN" --array=0-$((n_sizes - 1)) \
        --output="$logs/%x-%A_%a.out" slurm/train.sbatch)
    echo "students: job $train (array 0-$((n_sizes - 1)): ${TRAIN_SIZES[*]})"
    exit 0
fi

law=$(submit "${serial[@]}" --time="$TIME_LAW" --output="$logs/%x-%j.out" slurm/teacher_law.sbatch)
echo "law:      job $law"
shards=$(submit "${cpu[@]}" --time="$TIME_SHARD" --array=0-"$N_SHARDS" \
    --dependency=afterok:"$law" --output="$logs/%x-%A_%a.out" slurm/teacher_shards.sbatch)
echo "teacher:  job $shards (array 0-$N_SHARDS, the last task is the test set)"
merge=$(submit "${serial[@]}" --time="$TIME_MERGE" --dependency=afterok:"$shards" \
    --output="$logs/%x-%j.out" slurm/merge.sbatch)
echo "merge:    job $merge"
if [ "${1:-all}" = "strip" ]; then
    physics=$(submit "${cpu[@]}" --time="$TIME_PHYSICS" --dependency=afterok:"$merge"         --output="$logs/%x-%j.out" slurm/physics.sbatch)
    echo "physics:  job $physics (results in ${data}/physics)"
    echo
    echo "Follow with:  squeue -u $USER      logs in $logs"
    exit 0
fi
train=$(submit "${train_res[@]}" --time="$TIME_TRAIN" --array=0-$((n_sizes - 1)) \
    --dependency=afterok:"$merge" --output="$logs/%x-%A_%a.out" slurm/train.sbatch)
echo "students: job $train (array 0-$((n_sizes - 1)): ${TRAIN_SIZES[*]})"
echo
echo "Follow with:  squeue -u $USER      logs in $logs      results in ${data}/students"
