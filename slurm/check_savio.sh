#!/bin/bash
# Print what slurm/config.sh needs: accounts, partitions, QoS, GPU types, cores per node.
# Run on a Savio login node:  bash slurm/check_savio.sh
set -u

echo "=== Your accounts, partitions and QoS (copy a matching line into config.sh) ==="
sacctmgr -p show associations user="$USER" format=account,partition,qos | column -t -s '|'

echo
echo "=== Partitions: nodes, cores per node, GPUs (gres) ==="
sinfo -o "%15P %6D %6c %25G %10l" | sort -u

echo
echo "=== GPU partitions only ==="
sinfo -o "%15P %25G %6c" | grep -i gpu | sort -u

echo
echo "=== Python modules ==="
module avail python 2>&1 | grep -i python | head -20

echo
echo "=== Scratch ==="
ls -ld "/global/scratch/users/$USER" 2>/dev/null || echo "no /global/scratch/users/$USER"
