import time
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DataIterator:
    def __init__(self, dataloader, sampler=None, max_retries=5, retry_sleep=5):
        self.dataloader = dataloader
        self.sampler = sampler
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.loader = iter(dataloader)

    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def reset(self):
        self.loader = iter(self.dataloader)

    def next(self, epoch):
        iter_retries = 0
        iter_successful = False
        while not iter_successful:
            try:
                sample = next(self.loader)
                iter_successful = True
                return sample

            except StopIteration:
                logger.info("Exhausted dataset. Restarting loader...")
                self.set_epoch(epoch)
                self.reset()

            except Exception as e:
                if iter_retries < self.max_retries:
                    iter_retries += 1
                    logger.warning(
                        f"Data loading error retry {iter_retries}/{self.max_retries}: {e}"
                    )
                    time.sleep(self.retry_sleep)
                else:
                    raise RuntimeError(
                        f"Failed loading data after {self.max_retries} retries"
                    ) from e
                    
                    
def set_lr_para(
    lr0: float = 0.003,            # base LR from paper
    B0: int = 4096,               # original global batch size from paper
    B_global: int = 512 * 4,         # your current global batch size
):
    """
    Define parameter groups for optimizer with separate weight decays
    for head, early, mid, and late transformer blocks, but same learning rate
    scaled according to global batch size.

    Args:
        lr0 (float): base learning rate in the original paper
        B0 (int): original global batch size in the paper
        B_global (int): your current global batch size

    Returns:
        dict: dictionary containing hyperparameters
    """

    # Scale learning rate linearly based on global batch
    lr_scaled = lr0 * B_global / B0

    OPTIMIZER_PARAMS = {
        "lr_head": 1.5*lr_scaled,
        "lr_early": lr_scaled,
        "lr_mid": lr_scaled,
        "lr_late": lr_scaled,
        "weight_decay_head": 0.05,
        "weight_decay_early": 0.03,
        "weight_decay_mid": 0.03,
        "weight_decay_late": 0.03,
    }

    return OPTIMIZER_PARAMS


def create_optimizer(model, params: dict):
    param_groups = [
        {
            "params": model.head.parameters(),
            "lr": params["lr_head"],
            "weight_decay": params["weight_decay_head"],
            "initial_lr": params["lr_head"],  # important for cosine schedule
        },
    ]

    num_layers = len(model.transformer.layers)
    early_idx = list(range(0, int(0.4 * num_layers)))
    mid_idx   = list(range(int(0.4 * num_layers), int(0.7 * num_layers)))
    late_idx  = list(range(int(0.7 * num_layers), num_layers))

# Add transformer blocks as separate param groups
    param_groups += [
        {
            "params": [p for i, layer in enumerate(model.transformer.layers) if i in early_idx for p in layer.parameters()],
            "lr": params["lr_early"],
            "weight_decay": params["weight_decay_early"],
            "initial_lr": params["lr_early"],
        },
        {
            "params": [p for i, layer in enumerate(model.transformer.layers) if i in mid_idx for p in layer.parameters()],
            "lr": params["lr_mid"],
            "weight_decay": params["weight_decay_mid"],
            "initial_lr": params["lr_mid"],
        },
        {
            "params": [p for i, layer in enumerate(model.transformer.layers) if i in late_idx for p in layer.parameters()],
            "lr": params["lr_late"],
            "weight_decay": params["weight_decay_late"],
            "initial_lr": params["lr_late"],
        },
    ]
    optimizer = torch.optim.AdamW(param_groups)
    return optimizer

# --- Cosine decay + quadratic warmup schedule ---
def cosine_schedule(epoch, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
    """
    Quadratic warmup + cosine decay per param group.
    """
    for pg in optimizer.param_groups:
        base_lr = pg["initial_lr"]  # use the initial LR for each block
        if epoch < warmup_epochs:
            lr = min_lr + (base_lr - min_lr) * (epoch / warmup_epochs) ** 2
        else:
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
        pg["lr"] = lr
    return [pg["lr"] for pg in optimizer.param_groups]



    