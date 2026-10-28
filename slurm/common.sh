# Sourced by every job script: settings, environment, single-threaded BLAS.
source "${SLURM_SUBMIT_DIR:-.}/slurm/config.sh"
module load "$PYTHON_MODULE"
source "$VENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
DATA="${WORK}/${PHASE3_NAME}"
strip_flag=()
if [ "${STRIP:-0}" = "1" ]; then
    strip_flag=(--strip)
fi
mkdir -p "$DATA/shards" "$DATA/logs"

# Worker counts come from what SLURM granted. Not from `nproc`: GNU nproc
# returns OMP_NUM_THREADS when it is set, so with the line above it says 1 --
# the cause of the first runs using one core out of 32 or 40.
CORES=${SLURM_CPUS_PER_TASK:-$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)}
run() {
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        srun --ntasks=1 --cpus-per-task="$CORES" --cpu-bind=none "$@"
    else
        "$@"
    fi
}
visible=$(run env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
echo "job ${SLURM_JOB_ID:-local} on $(hostname), $CORES cores granted, $visible usable, $(date)"
