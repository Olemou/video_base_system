
import math
from typing import Iterator, Optional

import numpy as np
import torch

from src.src_utils import get_logger
from torch.utils.data import DistributedSampler, RandomSampler

logger = get_logger("WeightedSampler")


class DistributedWeightedSampler(DistributedSampler):
    """
    This class implements a weighted sampler for distributed training.
    See https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler for more details.

    It shares the same interface as `torch.utils.data.DistributedSampler`.
    The effective change is replacing `DistributedSampler`'s `torch.randperm` for generating the sequence
    of indices with `numpy.random.Generator.choice`, with replacement. This allows weighted sampling and
    avoiding issue with `torch.randperm` when the number of samples is larger than 2^24 samples.
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        logger.info(
            f"Using DistributedWeightedSampler with rank {rank} / {num_replicas}"
        )
        assert hasattr(
            dataset, "sample_weights"
        ), "Dataset must have sample_weights property for using DistributedWeightedSampler"
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    @property
    def sample_probabilities(self) -> np.ndarray:
        sample_weights = self.dataset.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.cpu().numpy()
        elif isinstance(sample_weights, list):
            sample_weights = np.array(sample_weights)
        assert isinstance(
            sample_weights, np.ndarray
        ), f"sample_weights must be a numpy array, torch.Tensor, or python list; got {type(sample_weights)}"
        return sample_weights / np.sum(sample_weights)

    def __iter__(self) -> Iterator:
        n = len(self.dataset)

        # deterministically shuffle based on epoch and seed
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.choice(
            range(0, n),
            size=self.total_size,
            p=self.sample_probabilities,
            replace=True,
        ).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)