#!/bin/bash
# One array task = one node solving SHARD_SIZE training instances with the shared law.
# The extra task (index N_SHARDS) solves the test instances.
#SBATCH --job-name=gw-teacher
#SBATCH --nodes=1
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source slurm/common.sh

task=${SLURM_ARRAY_TASK_ID}
if [ "$task" -lt "$N_SHARDS" ]; then
    first=$(( TRAIN_FIRST + task * SHARD_SIZE ))
    count=$SHARD_SIZE
    split=train
    name=$(printf "train_%03d" "$task")
else
    first=$TEST_FIRST
    count=$TEST_COUNT
    split=test
    name=test
fi

run python scenarios/phase3_dataset.py ${strip_flag[@]+"${strip_flag[@]}"} --theta "$DATA/theta.npy" \
    --first "$first" --count "$count" --split "$split" --name "$name" \
    --nlp-budget "$NLP_BUDGET" --workers "$CORES" --out "$DATA/shards"
