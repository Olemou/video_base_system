import os
import sys
import argparse
import json
import time
import socket
from pathlib import Path
from app.utils import DataIterator
from app.utils import  cosine_schedule, set_lr_para, create_optimizer
import os
from src.src_utils.logging import gpu_timer, CSVLogger
from src.src_utils.utils import AverageMeter
from src.loss_fn.loss import UncertaintyAwareLoss
import gc
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
import random
from src.src_utils.logging import get_logger
from src.datasets.data_manager import init_data
from src.datasets.utils.utils import get_base_path, get_path_sheets, load_config
from app.model import KalmanFormerNetVideoModel
from src.src_utils.vision_config import VisionConfig
warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = get_logger("DDP Training",force=True)

#=============================================================================
# -------------------------------
# Helper to set requires_grad
# -------------------------------
def set_trainable(module, flag: bool):
    """Set requires_grad for all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = flag

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
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Data loading workers per GPU")
    parser.add_argument("--which_dtype", type=str, default="float16",
                        help="Data type for training")
    parser.add_argument("--loss_reg_std_mult", type=float, default=3.0,
                        help="Standard deviation multiplier for loss regularization")
    parser.add_argument("--loss_reg_min_epoch", type=int, default=5,
                        help="Minimum epoch to start loss regularization")  
    parser.add_argument("--loss_reg_num_tracking_steps", type=int, default=300,
                        help="Number of steps to track for loss regularization")
    parser.add_argument("--save_every_freq", type=int, default=20,
                        help="Frequency (in epochs) to save checkpoints (0 to disable)")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of warmup epochs") 

    # Model and data
    
    parser.add_argument("--data_root", type=str, default="~/data",
                        help="Root directory for dataset (must be accessible from all nodes)")
    
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for checkpoints (shared across nodes)")
    
    parser.add_argument("--checkpoint_dir", type=str,
                        help="Directory for saving checkpoints (shared across nodes)")
    parser.add_argument("--interpreter_dir", type=str, default = "/content/drive/MyDrive/data",
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


# Training function
# -------------------------------
def main(
):

# -------------------------------
    # Setup =================================================================================================
    setup_environment()
    args = parse_arguments()
    args = init_distributed_mode(args)
    seed_everything(args.seed)
    
    # FIX: Verify distributed setup before proceeding
    if args.distributed:
        synchronize()
        if is_main_process():
            logger.info(f"✅ All {args.world_size} processes synchronized across {args.nnodes} nodes")

    #=====================================================================================================
    folder = args.interpreter_dir
    rank = args.rank
    which_dtype = args.which_dtype
    batch_size = args.global_batch_size // args.world_size
    world_size = args.world_size
    lr = args.learning_rate * (args.global_batch_size / 4096)  # scale LR by global batch size
    loss_reg_std_mult = args.loss_reg_std_mult
    loss_reg_min_epoch  = args.loss_reg_min_epoch
    loss_reg_num_tracking_steps = args.loss_reg_num_tracking_steps
    save_every_freq = args.save_every_freq
    
    scaler = torch.amp.GradScaler()
    sync_gc = True,
    GARBAGE_COLLECT_ITR_FREQ=50
    
    #================================================================================================  
    #Data Loading and Dataloader setup   
    #================================================================================================  
   
    BASE_DIR = Path(__file__).resolve().parents[1]
    path_config = BASE_DIR / "config" / "dataset_config.yaml"
    #==========================================================================================
    config_path = load_config(path_config)
    
    dataloader, sampler = init_data(
    data_paths = get_path_sheets(config_path),
    batch_size=16,
    num_workers=4,
    base_path=get_base_path(config_path),
    world_size=world_size,
    rank=rank,  
) 
    steps_per_epoch = len(dataloader) 
  #=========================================================
     #Intialization of model encoder
    #============  ================================================================
    model = KalmanFormerNetVideoModel(
     config = VisionConfig()
     ).to(args.device)
    #=====================================================================
    
    #================================##===================================================    
    """Train a transformer model with gradual unfreezing on epoch 0, full training afterward."""
    num_layers = len(model.attn_layers)
    early = list(range(0, int(0.4 * num_layers)))
    mid   = list(range(int(0.4 * num_layers), int(0.7 * num_layers)))
    late  = list(range(int(0.7 * num_layers), num_layers))
    #===============================================================================================
    #Average Metrics
    #====================================================================================
    loss_meter = AverageMeter()
    iter_time_meter = AverageMeter()
    gpu_time_meter = AverageMeter()
    data_elapsed_time_meter = AverageMeter()
    #===============================================================================================
    log_freq = 10
    CHECKPOINT_FREQ = 1
    MAX_REPEAT_COUNTS = 10
    #================================================================================
     # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pth.tar"
    latest_path = os.path.join(folder, latest_file)
    #================================================================================
   # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    data_iter = DataIterator(dataloader, sampler=sampler)
    
    #===============================================================================================
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False
    #================================================================================================
    
    #============================== Loss Initialisation here ===============================================  
    criterion = UncertaintyAwareLoss(prior_weight=0.5, TotalEpochs=args.num_epochs, temperature=0.1)    
    # =============================================================================
    
    #==================================================optimizer===================================
    parameters = set_lr_para(
    B_global= args.global_batch_size)
    optimizer = create_optimizer( model = model,params=  parameters)
    #===========================================================================================================
    trailing_losses = []
    step_count = 0
    # ----- Garbage collection before batch -----
    if sync_gc:
        gc.disable()
        gc.collect()

    for epoch in range(args.num_epochs):
        model.train()
        data_iter.set_epoch(epoch)
        trainable_layers = set()  # track unfreezed layers (epoch 0)
        
        # ================================================================
        logger.info("Training Start For Epoch %d" % (epoch + 1))
        #=================================================================================================
        loss_meter.reset()
        iter_time_meter.reset()
        gpu_time_meter.reset()
        #====================================================================
        #y cosine schedule at each epoch start
        cosine_schedule(epoch, optimizer, args.warmup_epochs, args.num_epochs, min_lr=1e-6)
        
        
        #===============================================================================================
        def save_checkpoint(epoch: int, path: str):
            if is_main_process():
                save_dict = {
                    "encoder": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "scaler": None if scaler is None else scaler.state_dict(),
                    "epoch": epoch,
                    "loss": loss_meter.avg,
                    "batch_size": batch_size,
                    "world_size": world_size,
                    "lr_encoder_ref": lr,
                    
                }
                try:
                    torch.save(save_dict, path)
                except Exception as e:
                    logger.info(f"Encountered exception when saving checkpoint: {e}")
            
        #======================#=#===========================================================
    
        for itr in range(steps_per_epoch):
            progress = itr / steps_per_epoch
            #==============#
            itr_start_time = time.time()
            #===========================
            # Fetch batch safely
            batch = data_iter.next(epoch)

            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
                inputs, targets = inputs.to(args.device), targets.to(args.device)
            else:
                inputs = batch.to(args.device)
                targets = None
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            # ---------- Layer freezing logic ----------
            if epoch == 0:
                # Gradual unfreezing for first epoch
                def gradual_unfreeze(layers, start_progress, end_progress, current_progress):
                    if current_progress < start_progress:
                        return
                    elif current_progress >= end_progress:
                        for l in layers:
                            if l not in trainable_layers:
                                set_trainable(model.attn_layers[l], True)
                                trainable_layers.add(l)
                    else:
                        frac = (current_progress - start_progress) / (end_progress - start_progress)
                        num_to_unfreeze = max(1, int(len(layers) * frac))
                        for l in layers[-num_to_unfreeze:]:
                            if l not in trainable_layers:
                                set_trainable(model.attn_layers[l], True)
                                trainable_layers.add(l)

                gradual_unfreeze(late, start_progress=0.0, end_progress=0.2, current_progress=progress)
                gradual_unfreeze(mid,  start_progress=0.2, end_progress=0.5, current_progress=progress)
                gradual_unfreeze(early,start_progress=0.5, end_progress=1.0, current_progress=progress)

                set_trainable(model.head, True)

            else:
                # Epoch >=1: all layers trainable
                for layer in model.attn_layers:
                    set_trainable(layer, True)
                set_trainable(model.head, True)

             # ----- Periodic garbage collection -----
            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()
            # ---------- Forward and backward ----------
            def train_forward_step():
                outputs = model(inputs)
                with torch.amp.autocast(dtype=dtype, enabled=mixed_precision,device_type=args.device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets, epoch) 
                
                # Step 2. Backward & step
                run_step = True
                if loss_reg_std_mult is not None and len(trailing_losses) > 0 :
                    meanval = np.mean(trailing_losses)
                    stdval = np.std(trailing_losses)
                    max_bound = meanval + loss_reg_std_mult * stdval
                    if (loss > max_bound and epoch > loss_reg_min_epoch and len(trailing_losses)> int(0.5 * loss_reg_num_tracking_steps)):
                        run_step = False
                        loss.backward()
                        logger.info(
                                f"Loss {loss} is above bound {meanval} + {loss_reg_std_mult} * {stdval}. Skipping step."
                            )
                    if run_step:
                        if mixed_precision:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                        else:
                            loss.backward()
                        if mixed_precision:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    optimizer.zero_grad()

                return  (
                    float(loss),
                    run_step,
                )
            (loss,run_step), gpu_etime_ms = gpu_timer(train_forward_step)
            loss_meter.update(loss, n=inputs.size(0))
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)
            #=================================================================================
            if loss_reg_std_mult is not None:
                if run_step:
                    trailing_losses.append(loss)
                    if len(trailing_losses) > loss_reg_num_tracking_steps:
                        trailing_losses = trailing_losses[1:]
                else:
                    step_count += 1
                    if step_count > MAX_REPEAT_COUNTS:
                        raise RuntimeError(
                            "Loss is above bound for too many tries. Exiting."
                        )
            #===================================================================================== # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    gpu_etime_ms,
                    data_elapsed_time_ms,
                )
                if (
                    (itr % log_freq == 0)
                    or (itr == itr - 1)
                    or np.isnan(loss)
                    or np.isinf(loss)
                    ):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2f MB] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            optimizer.param_groups[0]["weight_decay"],
                            optimizer.param_groups[0]["lr"],
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )
            log_stats() 
            #=====================================================================================================
            
         # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == (args.num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_every_file = f"e{epoch}.pth.tar"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)

        if sync_gc:
            gc.enable()
