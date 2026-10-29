#!/bin/bash
# Submit the whole phase 3 pipeline on Savio, chained by dependencies:
#   law (1 node) -> teacher shards (N_SHARDS + 1 nodes) -> merge -> students (1 GPU per size)
# From the repository root on a login node:
#   bash slurm/submit_all.sh            # everything
#   bash slurm/submit_all.sh train      # only the students (dataset already merged)
#   bash slurm/submit_all.sh latent     # only the latent GA (dataset already merged)
#   bash slurm/submit_all.sh physics    # only the latent GA with the physics decoder
#   bash slurm/submit_all.sh strip      # strip-theory wing: law -> shards -> merge -> physics,
#                                       # in $WORK/phase3_strip (or $PHASE3_NAME)
#   bash slurm/submit_all.sh sensitivity  # SB-XC polar +-20 % on the strip comparison
#   bash slurm/submit_all.sh closedloop   # phase 4: closed loop, turbulence, wrong belief
#                                       # (on the Savio strip run, $WORK/phase3_strip_v2)
#                                       # (needs the strip dataset and command VAE in $WORK/phase3_strip)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-all}" = "strip" ] || [ "${1:-all}" = "sensitivity" ]; then
    # Another folder keeps an earlier strip run:  PHASE3_NAME=phase3_strip_v2 bash slurm/submit_all.sh strip
    export STRIP=1 PHASE3_NAME=${PHASE3_NAME:-phase3_strip}
fi
if [ "${1:-all}" = "closedloop" ]; then
    export STRIP=1 PHASE3_NAME=${PHASE3_NAME:-phase3_strip_v2}
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
if [ "${1:-all}" = "sensitivity" ] || [ "${1:-all}" = "closedloop" ]; then
    for f in dataset.npz theta.npy physics/command_vae_d16.pt; do
        if [ ! -f "$data/$f" ]; then
            echo "missing $data/$f: copy the strip dataset and its command VAE there first" >&2
            exit 1
        fi
    done
    if [ "$1" = "closedloop" ]; then
        job=$(submit "${cpu[@]}" --time="$TIME_CLOSEDLOOP" --output="$logs/%x-%j.out" slurm/closedloop.sbatch)
        echo "closed loop: job $job (results in ${data}/closedloop)"
    else
        job=$(submit "${cpu[@]}" --time="$TIME_SENSITIVITY" --output="$logs/%x-%j.out" slurm/sensitivity.sbatch)
        echo "sensitivity: job $job (results in ${data}/sensitivity)"
    fi
    exit 0
fi
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
