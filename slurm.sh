#!/bin/bash
#SBATCH --job-name=felix_multinode_ddp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1         # ONE torchrun per node
#SBATCH --gpus-per-node=h100:2       # 2 GPUs per node
#SBATCH --cpus-per-task=8
#SBATCH --mem=32768M
#SBATCH --partition=gpubase_bygpu_b3

# =========================
# MODULES
# =========================
module load StdEnv/2023
module load cuda/12.8

# =========================
# VENV
# =========================
source ~/video_base_system/venv/bin/activate

which python
python -c "import torch; print(torch.__version__)"

mkdir -p logs

# =========================
# MULTI-NODE CONFIG
# =========================
# Compute Canada pattern for master address
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=3456  # Use official port
export TORCH_NCCL_ASYNC_HANDLING=1

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# Get number of GPUs


# =========================
# PATHS
# =========================
export HOME=/home/olemou
EVAL_DIR="$HOME/shared/eval_dir"
CHECKPOINT_DIR="$HOME/shared/checkpoints/$SLURM_JOB_ID"
INTERPRETER_DIR="$HOME/shared/logfiles/$SLURM_JOB_ID"
MONITORING_DIR="$HOME/shared/monitoring/$SLURM_JOB_ID"

if [ $SLURM_NODEID -eq 0 ]; then
    mkdir -p "$CHECKPOINT_DIR" "$INTERPRETER_DIR" "$MONITORING_DIR" "$EVAL_DIR"
fi

# Wait a few seconds for the head node to create directories
sleep 5

# =========================
# LAUNCH TORCHRUN
# =========================
# Extract the number of GPUs from SLURM_GPUS_PER_NODE (e.g., "h100:2" -> 2)
NGPUS=$(echo $SLURM_GPUS_PER_NODE | cut -d: -f2)

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$NGPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    launch.py \
    --is_distributed \
    --eval_dir "$EVAL_DIR" \
    --monitoring_dir "$MONITORING_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --interpreter_dir "$INTERPRETER_DIR" \
    --batch_size 32 \
    --num_epochs 100