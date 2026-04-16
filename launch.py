import time
import torch
from app.utils import DataIterator
from app.utils import  cosine_schedule
import logging
import os
from src.src_utils.logging import gpu_timer, CSVLogger
from src.src_utils.utils import AverageMeter
import gc
import numpy as np

logger = logging.getLogger(__name__)

# -------------------------------
# Helper to set requires_grad
# -------------------------------
def set_trainable(module, flag: bool):
    """Set requires_grad for all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = flag

# -------------------------------
# Training function
# -------------------------------
def train_model(
    model,
    dataloader,
    criterion,
    device,
    num_epochs,
    args,
    steps_per_epoch,
    optimizer,
    warmup_epochs,
    scaler = torch.amp.GradScaler(),
    sampler=None,
    sync_gc = True,
    GARBAGE_COLLECT_ITR_FREQ=50
):
    """Train a transformer model with gradual unfreezing on epoch 0, full training afterward."""
    num_layers = len(model.transformer.layers)
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
    folder = args.interpreter_dir
    rank = args.rank
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
    which_dtype = args.dtype
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
    batch_size = args.global_batch_size // args.world_size
    world_size = args.world_size
    lr = args.learning_rate * (args.global_batch_size / 4096)  # scale LR by global batch size
    #============  ================================================================
    loss_reg_std_mult = args.loss_reg_std_mult
    loss_reg_min_epoch  = args.loss_reg_min_epoch
    loss_reg_num_tracking_steps = args.loss_reg_num_tracking_steps
    save_every_freq = args.save_every_freq
    #=====================================================================
    trailing_losses = []
    step_count = 0
    # ----- Garbage collection before batch -----
    if sync_gc:
        gc.disable()
        gc.collect()

    for epoch in range(num_epochs):
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
        cosine_schedule(epoch, optimizer, warmup_epochs, num_epochs, min_lr=1e-6)
        
        
        #===============================================================================================
        def save_checkpoint(epoch, path):
            if rank != 0:
                return
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
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                inputs = batch.to(device)
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
                                set_trainable(model.transformer.layers[l], True)
                                trainable_layers.add(l)
                    else:
                        frac = (current_progress - start_progress) / (end_progress - start_progress)
                        num_to_unfreeze = max(1, int(len(layers) * frac))
                        for l in layers[-num_to_unfreeze:]:
                            if l not in trainable_layers:
                                set_trainable(model.transformer.layers[l], True)
                                trainable_layers.add(l)

                gradual_unfreeze(late, start_progress=0.0, end_progress=0.2, current_progress=progress)
                gradual_unfreeze(mid,  start_progress=0.2, end_progress=0.5, current_progress=progress)
                gradual_unfreeze(early,start_progress=0.5, end_progress=1.0, current_progress=progress)

                set_trainable(model.head, True)

            else:
                # Epoch >=1: all layers trainable
                for layer in model.transformer.layers:
                    set_trainable(layer, True)
                set_trainable(model.head, True)

             # ----- Periodic garbage collection -----
            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()
            # ---------- Forward and backward ----------
            
            def train_forward_step():
                outputs = model(inputs)
                with torch.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) 
                
                # Step 2. Backward & step
                run_step = True
                if loss_reg_std_mult is not None:
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
            loss_meter.update(loss.item(), n=inputs.size(0))
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
        if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_every_file = f"e{epoch}.pth.tar"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)

        if sync_gc:
            gc.enable()