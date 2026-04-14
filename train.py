import time
import torch
import logging

logger = logging.getLogger(__name__)

class RobustDataIterator:
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

    def next(self):
        retries = 0

        while True:
            try:
                return next(self.loader)

            except StopIteration:
                # restart dataset
                self.reset()
                continue

            except Exception as e:
                if retries < self.max_retries:
                    retries += 1
                    logger.warning(f"Retry {retries}/{self.max_retries}: {e}")
                    time.sleep(self.retry_sleep)
                else:
                    raise RuntimeError("Data loading failed") from e
                
                
                
    for epoch in range(start_epoch, num_epochs):
    logger.info(f"Epoch {epoch + 1}")

    # meters (logging)
    loss_meter = AverageMeter()
    mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
    iter_time_meter = AverageMeter()
    gpu_time_meter = AverageMeter()
    data_elapsed_time_meter = AverageMeter()

    # IMPORTANT: reset epoch sampling
    unsupervised_sampler.set_epoch(epoch)

    # restart dataloader each epoch
    loader = iter(unsupervised_loader)

    for itr in range(ipe):
        itr_start_time = time.time()

        iter_successful = False
        iter_retries = 0
        NUM_RETRIES = 5

        while not iter_successful:
            try:
                # -------------------------
                # GET BATCH
                # -------------------------
                sample = next(loader)
                iter_successful = True

            except StopIteration:
                # -------------------------
                # DATASET ENDED EARLY → RESTART
                # -------------------------
                logger.info("Exhausted dataset. Restarting loader...")

                unsupervised_sampler.set_epoch(epoch)
                loader = iter(unsupervised_loader)

            except Exception as e:
                # -------------------------
                # TEMPORARY DATA ERROR → RETRY
                # -------------------------
                if iter_retries < NUM_RETRIES:
                    iter_retries += 1
                    logger.warning(
                        f"Data loading error retry {iter_retries}/{NUM_RETRIES}: {e}"
                    )
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        f"Failed loading data after {NUM_RETRIES} retries"
                    ) from e

        # -------------------------
        # TRAIN STEP (YOUR MODEL HERE)
        # -------------------------
        # output = model(sample)
        # loss = criterion(output)
        # loss.backward()
        # optimizer.step()

        itr_time = time.time() - itr_start_time
        iter_time_meter.update(itr_time)
        
        
        
        
        import time
import torch
import logging

logger = logging.getLogger(__name__)


class RobustDataIterator:
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

    def next(self):
        retries = 0

        while True:
            try:
                return next(self.loader)

            except StopIteration:
                self.reset()
                continue

            except Exception as e:
                if retries < self.max_retries:
                    retries += 1
                    logger.warning(f"Retry {retries}/{self.max_retries}: {e}")
                    time.sleep(self.retry_sleep)
                else:
                    raise RuntimeError("Data loading failed") from e


class Trainer:
    def __init__(self, model, optimizer, criterion,
                 train_loader, eval_loader, sampler,
                 device, save_path="checkpoint.pth"):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.sampler = sampler

        self.save_path = save_path
        self.data_iter = RobustDataIterator(train_loader, sampler)

    # ---------------- SAVE ----------------
    def save(self, epoch):
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, self.save_path)

        logger.info(f"Saved checkpoint at epoch {epoch}")

    # ---------------- LOAD ----------------
    def load(self):
        try:
            ckpt = torch.load(self.save_path)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            logger.info(f"Resumed from epoch {ckpt['epoch']}")
            return ckpt["epoch"]
        except:
            logger.info("No checkpoint found. Starting fresh.")
            return 0

    # ---------------- EVAL ----------------
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.eval_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.criterion(out)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.eval_loader)
        logger.info(f"[EVAL] loss = {avg_loss:.4f}")
        self.model.train()
        return avg_loss

    # ---------------- ONE EPOCH ----------------
    def train_one_epoch(self, epoch, ipe):
        logger.info(f"Epoch {epoch + 1}")

        self.data_iter.set_epoch(epoch)

        loss_sum = 0.0

        for itr in range(ipe):

            batch = self.data_iter.next().to(self.device)

            self.optimizer.zero_grad()

            out = self.model(batch)
            loss = self.criterion(out)

            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()

        avg_loss = loss_sum / ipe
        logger.info(f"Epoch {epoch + 1} loss = {avg_loss:.4f}")

        return avg_loss

    # ---------------- FULL TRAIN ----------------
    def train(self, start_epoch, num_epochs, ipe,
              eval_every=1, save_every=1):

        for epoch in range(start_epoch, num_epochs):

            self.train_one_epoch(epoch, ipe)

            if epoch % eval_every == 0:
                self.evaluate()

            if epoch % save_every == 0:
                self.save(epoch)