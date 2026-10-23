#!/bin/bash
# Scaling study: one array task per training-set size in TRAIN_SIZES, on one GPU
# or one CPU node (TRAIN_ON in config.sh).
# Each task trains both students, evaluates them on the test instances and writes
# $DATA/students/n_<size>/ (summary, figure, weights).
#SBATCH --job-name=gw-train
#SBATCH --nodes=1
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source slurm/common.sh
device=cpu
if [ "$TRAIN_ON" = "gpu" ]; then
    device=cuda
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
fi

size=${TRAIN_SIZES[$SLURM_ARRAY_TASK_ID]}
max_train=()
if [ "$size" != "all" ]; then
    max_train=(--max-train "$size")
fi

python scenarios/phase3_student.py --data "$DATA/dataset.npz" --device "$device" \
    --steps "$TRAIN_STEPS" ${max_train[@]+"${max_train[@]}"} --threads "$(nproc)" \
    --out "$DATA/students" --tag "n_${size}"
