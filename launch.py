import time
import torch
import logging

logger = logging.getLogger(__name__)

# -------------------------------
# DataIterator class
# -------------------------------
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
    optimizer,
    criterion,
    device,
    num_epochs,
    steps_per_epoch,
    sampler=None,
):
    """Train a transformer model with gradual unfreezing on epoch 0, full training afterward."""
    num_layers = len(model.transformer.layers)
    early = list(range(0, int(0.4 * num_layers)))
    mid   = list(range(int(0.4 * num_layers), int(0.7 * num_layers)))
    late  = list(range(int(0.7 * num_layers), num_layers))

    data_iter = DataIterator(dataloader, sampler=sampler)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        data_iter.set_epoch(epoch)
        trainable_layers = set()  # track unfreezed layers (epoch 0)

        for step in range(steps_per_epoch):
            progress = step / steps_per_epoch

            # Fetch batch safely
            batch = data_iter.next(epoch)

            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                inputs = batch.to(device)
                targets = None

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

            # ---------- Forward and backward ----------
            outputs = model(inputs)
            loss = criterion(outputs, targets) if targets is not None else outputs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Logging
            if (step + 1) % 50 == 0 or step == steps_per_epoch - 1:
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{steps_per_epoch}], "
                    f"Loss: {loss.item():.4f}, Progress: {progress:.2f}"
                )

        avg_loss = epoch_loss / steps_per_epoch
        logger.info(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")