#!/bin/bash
# Submit the held-out cross-country check (slurm/heldout_xc.sbatch) on the strip-wing run.
set -euo pipefail
export STRIP=1 PHASE3_NAME=${PHASE3_NAME:-phase3_strip_v2}
source slurm/config.sh
for f in physics/command_vae_d16.pt dagger/distill/distilled_flow.pt; do
    [ -f "$WORK/$PHASE3_NAME/$f" ] || { echo "missing $WORK/$PHASE3_NAME/$f" >&2; exit 1; }
done
mkdir -p "$WORK/$PHASE3_NAME/logs"
sbatch --parsable --chdir="$PWD" --account="$ACCOUNT" --partition="$CPU_PARTITION" --qos="$CPU_QOS" \
    --cpus-per-task="$CPU_CORES" --time="$TIME_DAGGER" --output="$WORK/$PHASE3_NAME/logs/%x-%j.out" \
    --export=ALL slurm/heldout_xc.sbatch
echo "results in $WORK/$PHASE3_NAME/dagger/crosscountry_heldout/"
