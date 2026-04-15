import time
import torch
import logging

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

        while True:
            try:
                sample = next(self.loader)
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
    