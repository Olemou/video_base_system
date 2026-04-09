#!/bin/bash
#SBATCH --job-name=felix_multinode_ddp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=2                   # 2 nodes
#SBATCH --ntasks-per-node=2         # 2 tasks per node → 1 task per GPU
#SBATCH --gpus-per-node=h100:2      # 2 full H100 GPUs per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --mem=64G
#SBATCH --partition=gpubase_interac   # adjust to your H100 partition

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR=~/video_base_system            # CHANGE THIS
VENV_PATH="${PROJECT_DIR}/venv"
REQUIREMENTS_FILE="${PROJECT_DIR}/requirements.txt"
VENV_FLAG_FILE="${VENV_PATH}/.setup_complete"

# =============================================================================
# VIRTUAL ENVIRONMENT SETUP WITH VERIFICATION
# =============================================================================

setup_venv() {
    local need_install=0
    
    # Check if venv exists
    if [ ! -d "$VENV_PATH" ]; then
        echo "📦 Virtual environment not found at: $VENV_PATH"
        need_install=1
    # Check if flag file exists (indicates successful installation)
    elif [ ! -f "$VENV_FLAG_FILE" ]; then
        echo "⚠️  Virtual environment exists but installation flag missing. Reinstalling..."
        need_install=1
    # Verify Python executable
    elif [ ! -f "$VENV_PATH/bin/python" ]; then
        echo "⚠️  Virtual environment corrupted (no Python binary). Recreating..."
        rm -rf "$VENV_PATH"
        need_install=1
    else
        echo "✅ Virtual environment found and validated"
    fi
    
    # Create/update venv if needed
    if [ $need_install -eq 1 ]; then
        echo "🔨 Creating fresh virtual environment..."
        python3 -m venv "$VENV_PATH" --clear
        
        echo "📥 Installing packages from requirements.txt..."
        source "$VENV_PATH/bin/activate"
        
        # Upgrade pip
        pip install --upgrade pip setuptools wheel
        
        # Install requirements
        if [ -f "$REQUIREMENTS_FILE" ]; then
            pip install -r "$REQUIREMENTS_FILE"
            
            # Create flag file to indicate successful installation
            touch "$VENV_FLAG_FILE"
            echo "✅ Installation complete. Flag file created."
        else
            echo "❌ ERROR: requirements.txt not found at $REQUIREMENTS_FILE"
            exit 1
        fi
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Export venv path for srun
    export VENV_PATH
}

# =============================================================================
# VERIFY INSTALLATION
# =============================================================================

verify_installation() {
    echo "🔍 Verifying installation..."
    
    # Check critical packages
    python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || exit 1
    python -c "import torchvision; print(f'✓ TorchVision {torchvision.__version__}')" || exit 1
    python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" || exit 1
    
    # Check CUDA
    if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        cuda_version=$(python -c "import torch; print(torch.version.cuda)")
        echo "✓ CUDA $cuda_version available"
    else
        echo "❌ ERROR: CUDA not available"
        exit 1
    fi
    
    echo "✅ All packages verified"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "=========================================="
echo "🔧 Loading modules..."
echo "=========================================="

# Load required modules
module load cuda/12.6
module load python/3.12

# Setup and activate virtual environment
setup_venv
verify_installation

# Create log directory
mkdir -p logs

# =============================================================================
# MULTI-NODE CONFIGURATION
# =============================================================================

# Get node information
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
master_addr=${nodes[0]}
master_port=29500

echo "=========================================="
echo "🌐 MULTI-NODE TRAINING SETUP"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Total Nodes: $SLURM_NNODES"
echo "Node List: ${nodes[@]}"
echo "Master Node: $master_addr"
echo "Current Node: $(hostname) (Rank: $SLURM_NODEID)"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "Python: $(which python)"
echo "Virtual Env: $VENV_PATH"
echo "=========================================="

# Shared paths
DATA_ROOT="~/data"  # CHANGE THIS
CHECKPOINT_DIR="~shared/checkpoints/$SLURM_JOB_ID"
INTERPRETER_DIR="~shared/interpreter/$SLURM_JOB_ID"
OUTPUT_DIR="~shared/output/$SLURM_JOB_ID"

# Master node creates directories
if [ $SLURM_NODEID -eq 0 ]; then
    mkdir -p $CHECKPOINT_DIR
    mkdir -p $INTERPRETER_DIR
    mkdir -p $OUTPUT_DIR
    echo "📁 Master node created: $CHECKPOINT_DIR"
    echo "📁 Master node created: $INTERPRETER_DIR"
    echo "📁 Master node created: $OUTPUT_DIR"
fi
sleep 5

# =============================================================================
# LAUNCH TRAINING
# =============================================================================

echo "🚀 Launching multi-node training..."

# Create a wrapper script to ensure venv is activated on all nodes
cat > /tmp/run_train_${SLURM_JOB_ID}.sh << EOF
#!/bin/bash
source "$VENV_PATH/bin/activate"
torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --master_addr=$master_addr \
    --master_port=$master_port \
    train.py \
    --is_distributed \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --interpreter_dir $INTERPRETER_DIR \
    --batch_size 512 \
    --num_epochs 100
EOF

chmod +x /tmp/run_train_${SLURM_JOB_ID}.sh

# Launch with srun using the wrapper
srun /tmp/run_train_${SLURM_JOB_ID}.sh

# Cleanup
rm -f /tmp/run_train_${SLURM_JOB_ID}.sh

echo "=========================================="
echo "✅ Multi-node training completed for job $SLURM_JOB_ID"
echo "=========================================="