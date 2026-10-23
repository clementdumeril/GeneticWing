# Sourced by every job script: settings, environment, single-threaded BLAS.
source "${SLURM_SUBMIT_DIR:-.}/slurm/config.sh"
module load "$PYTHON_MODULE"
source "$VENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
DATA="${WORK}/phase3"
mkdir -p "$DATA/shards" "$DATA/logs"

# On Savio the batch script itself stays pinned to one core even when the job
# holds more: only a job step started by srun sees them all. So every program
# runs through `run`, and CORES comes from what SLURM granted, not from nproc.
CORES=${SLURM_CPUS_PER_TASK:-$(nproc)}
run() {
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        srun --ntasks=1 --cpus-per-task="$CORES" --cpu-bind=none "$@"
    else
        "$@"
    fi
}
echo "job ${SLURM_JOB_ID:-local} on $(hostname), $CORES cores granted, $(run nproc) visible to programs, $(date)"
