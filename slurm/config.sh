# GeneticWing on Savio: the only file to edit.
# Fill the values from the output of  bash slurm/check_savio.sh  (run on a login node).

# --- Account and partitions -------------------------------------------------
ACCOUNT="ic_piml"

# CPU nodes, for the teacher (whole nodes, all cores used by one job).
# Savio bills the whole node, but SLURM only gives the job the cores it asks
# for: without CPU_CORES the job sees 1 core and runs 32 times slower.
CPU_PARTITION="savio3"
CPU_QOS="savio_normal"
CPU_CORES=32                     # cores per savio3 node (check_savio.sh, CPUS column)

# GPU nodes, for the students. The QoS and the GPU type go together on Savio:
# take a matching pair from check_savio.sh (examples: A40 / a40_gpu3_normal,
# GTX2080TI / gtx2080_gpu3_normal). CPUs per GPU are fixed per GPU type.
GPU_PARTITION="savio3_gpu"
GPU_QOS="a40_gpu3_ica"
GPU_GRES="gpu:A40:1"
GPU_CPUS=8

# --- Software ---------------------------------------------------------------
PYTHON_MODULE="python/3.11.6-gcc-11.4.0"
WORK="/global/scratch/users/${USER}/geneticwing"   # venv, data and results live here
VENV="${WORK}/venv"

# --- Experiment size --------------------------------------------------------
# Training instances are ids TRAIN_FIRST .. TRAIN_FIRST + N_SHARDS * SHARD_SIZE - 1;
# the test set is ids 600..679, the same as the laptop run.
TRAIN_FIRST=1000
N_SHARDS=20                      # array tasks (one node each)
SHARD_SIZE=750                   # instances per task: 20 x 750 = 15k instances, about 30k solutions
TEST_FIRST=600
TEST_COUNT=80
NLP_BUDGET=6000

# Scaling study: number of training instances per student run ("all" = everything).
TRAIN_SIZES=(300 1000 3000 10000 all)
TRAIN_STEPS=40000

# Walltimes (hh:mm:ss). A laptop core solves one instance in about 18 s;
# 750 instances on a 32-core node take about 8 minutes, so these are generous.
TIME_LAW="00:30:00"
TIME_SHARD="02:00:00"
TIME_MERGE="00:20:00"
TIME_TRAIN="04:00:00"
