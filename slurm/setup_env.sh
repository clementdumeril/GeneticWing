#!/bin/bash
# One-time setup on a Savio login node, from the repository root:
#   bash slurm/setup_env.sh
# Creates a virtual environment in scratch and installs GeneticWing with PyTorch.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/config.sh

module load "$PYTHON_MODULE"
python3 --version
mkdir -p "$WORK"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip
# The default PyTorch wheel on Linux ships with CUDA; it runs on CPU nodes too.
python -m pip install -e ".[dev]"
python -c "import geneticwing, torch, scipy; print('geneticwing OK, torch', torch.__version__)"
python -m pytest -q -x tests/test_bernstein.py tests/test_aircraft_and_flatness.py
echo "Environment ready in $VENV"
