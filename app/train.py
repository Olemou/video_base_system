import os
import sys
import argparse
import json
import time
import socket
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models
import numpy as np
import random
from src.src_utils.logging import get_logger

logger = get_logger("DDP Training", log_dir="~/log", log_file="ddp_training.txt")

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment():
    """Configure environment variables for multi-node training."""
    
    # Optimize CPU threading
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    
    # NCCL optimizations for multi-node
    os.environ.setdefault("NCCL_DEBUG", "WARN")  # Changed from INFO to WARN
    os.environ.setdefault("NCCL_IB_DISABLE", "0")  # Enable InfiniBand if available
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "^docker0,lo")  # Exclude virtual interfaces
    os.environ.setdefault("NCCL_IB_TIMEOUT", "22")  # InfiniBand timeout
    os.environ.setdefault("NCCL_IB_RETRY_CNT", "7")  # Retry count
    
    # PyTorch optimizations
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    
    # Prevent tokenizer parallelism warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility across all nodes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments for multi-node training."""
    
    parser = argparse.ArgumentParser(
        description="Multi-Node Distributed Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Distributed arguments (usually set by torchrun)
    parser.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", 1)),
                        help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=int(os.environ.get("NODE_RANK", 0)),
                        help="Rank of current node")
    parser.add_argument("--nproc_per_node", type=int, 
                        default=int(os.environ.get("NPROC_PER_NODE", torch.cuda.device_count())),
                        help="Number of processes per node")
    parser.add_argument("--master_addr", type=str, 
                        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
                        help="Master node IP address")
    parser.add_argument("--master_port", type=int, 
                        default=int(os.environ.get("MASTER_PORT", 29500)),
                        help="Master node port")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Data loading workers per GPU")
    
    # Model and data
    
    parser.add_argument("--data_root", type=str, default="~/data",
                        help="Root directory for dataset (must be accessible from all nodes)")
    
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for checkpoints (shared across nodes)")
    
    parser.add_argument("--checkpoint_dir", type=str,
                        help="Directory for saving checkpoints (shared across nodes)")
    parser.add_argument("--interpreter_dir", type=str,
                        help="Directory for saving interpreter logs (shared across nodes)")

    # Checkpointing and logging
    parser.add_argument("--checkpoint_freq", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="Log every N iterations")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--is_distributed", action="store_true",
                        help="Enable distributed training")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use Automatic Mixed Precision")
    
    args = parser.parse_args()
    
    # Calculate effective batch size
    args.global_batch_size = args.batch_size * args.nnodes * args.nproc_per_node
    
    return args


# =============================================================================
# DISTRIBUTED INITIALIZATION
# =============================================================================

def init_distributed_mode(args):
    """Initialize distributed training for multiple nodes."""
    
    # Check if running under torchrun
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print(f"🖥️  Running in single GPU mode on {socket.gethostname()}")
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        args.distributed = False
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return args
    
    args.distributed = True
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    
    # FIX: Properly set node_rank from environment
    args.node_rank = int(os.environ.get("GROUP_RANK", args.node_rank))
    
    # Set device
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device(f"cuda:{args.local_rank}")
    
    # Initialize process group with timeout
    timeout = timedelta(minutes=30)
    
    # FIX: Better error handling for multi-node initialization
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            world_size=args.world_size,
            rank=args.rank,
            timeout=timeout
        )
    except Exception as e:
        logger.info(30*"=")
        logger.error(f"Failed to initialize process group: {e}")
        logger.error(f"Master: {args.master_addr}:{args.master_port}")
        logger.error(f"Rank: {args.rank}/{args.world_size}")
        sys.exit(1)
    
    # Synchronize all processes
    dist.barrier()
    
    # Verify all processes are connected
    if args.rank == 0:
        logger.info("\n" + "="*80)
        logger.info("🌐 MULTI-NODE DISTRIBUTED TRAINING INITIALIZED")
        logger.info("="*80)
        logger.info(f"Total Nodes: {args.nnodes}")
        logger.info(f"GPUs per Node: {args.nproc_per_node}")
        logger.info(f"Total Processes: {args.world_size}")
        logger.info(f"Global Batch Size: {args.global_batch_size}")
        logger.info(f"Master Address: {args.master_addr}:{args.master_port}")
        logger.info(f"Backend: nccl")
        logger.info("="*80 + "\n")
    
    # FIX: Proper node identification
    logger.info(f"📍 Node {args.node_rank} | Host: {socket.gethostname()} | "
                f"Rank: {args.rank}/{args.world_size} | GPU: {args.local_rank} | "
                f"PID: {os.getpid()}")
    
    # FIX: Add tensor synchronization test to verify multi-node communication
    if args.rank == 0:
        test_tensor = torch.ones(1).to(args.device)
    else:
        test_tensor = torch.zeros(1).to(args.device)
    
    dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
    
    if args.rank == 0:
        expected_sum = args.world_size
        actual_sum = test_tensor.item()
        if abs(actual_sum - expected_sum) < 0.001:
            logger.info(f"✅ Multi-node communication test passed (sum={actual_sum})")
            logger.info(30*"=" + "\n")
        else:
            logger.warning(f"⚠️ Communication test: expected {expected_sum}, got {actual_sum}")
    
    dist.barrier()
    
    return args


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_available() and dist.is_initialized():
        # FIX: Add barrier before cleanup
        dist.barrier()
        dist.destroy_process_group()
        logger.info(f"🧹 Rank {os.environ.get('RANK', 0)}: Distributed cleanup complete")
        logger.info(30*"=" + "\n")
      

# =============================================================================
# UTILITY FUNCTIONS FOR MULTI-NODE
# =============================================================================

def is_main_process():
    """Check if current process is the main process (rank 0)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes (average)."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def gather_tensors(tensor):
    """Gather tensors from all processes."""
    tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def synchronize():
    """Synchronize all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# =============================================================================
# MAIN FUNCTION EXAMPLE
# =============================================================================

def main():
    """Main training function."""
    
    # Setup
    setup_environment()
    args = parse_arguments()
    args = init_distributed_mode(args)
    seed_everything(args.seed)
    
    # FIX: Verify distributed setup before proceeding
    if args.distributed:
        synchronize()
        if is_main_process():
            logger.info(f"✅ All {args.world_size} processes synchronized across {args.nnodes} nodes")
    # Your training code here...
    print(f"Process {args.rank} ready for training")
    
    # Cleanup
    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()