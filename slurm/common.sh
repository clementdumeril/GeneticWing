# Sourced by every job script: settings, environment, single-threaded BLAS.
source "${SLURM_SUBMIT_DIR:-.}/slurm/config.sh"
module load "$PYTHON_MODULE"
source "$VENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
DATA="${WORK}/phase3"
mkdir -p "$DATA/shards" "$DATA/logs"
echo "job ${SLURM_JOB_ID:-local} on $(hostname), $(nproc) cores, $(date)"
