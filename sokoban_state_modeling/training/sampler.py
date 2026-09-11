"""Checkpointable sampler that visits every task before padding repeats."""

from __future__ import annotations

import math
from typing import Iterator, Sized

import numpy as np
try:
    from torch.utils.data import Sampler
except ImportError:  # Data generation/evaluation can run without the training stack.
    class Sampler:  # type: ignore[no-redef]
        def __class_getitem__(cls, _item):
            return cls


class FullCoverageSampler(Sampler[int]):
    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        seed: int = 0,
        full_coverage_cycle: bool = True,
        pad_final_batch: bool = True,
    ):
        if not full_coverage_cycle:
            raise ValueError("FullCoverageSampler requires full_coverage_cycle=true")
        if not pad_final_batch:
            raise ValueError("FullCoverageSampler requires pad_final_batch=true")
        if len(data_source) <= 0 or int(batch_size) <= 0:
            raise ValueError("data_source and batch_size must be non-empty/positive")
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.cycle = 0
        self.position = 0
        self._permutation: list[int] | None = None

    @property
    def padded_length(self) -> int:
        return math.ceil(len(self.data_source) / self.batch_size) * self.batch_size

    @property
    def padding_count(self) -> int:
        return self.padded_length - len(self.data_source)

    def _indices(self) -> list[int]:
        if self._permutation is None:
            rng = np.random.default_rng(self.seed + self.cycle)
            unique = rng.permutation(len(self.data_source)).tolist()
            self._permutation = unique + unique[: self.padding_count]
        return self._permutation

    def __iter__(self) -> Iterator[int]:
        indices = self._indices()
        while self.position < len(indices):
            value = indices[self.position]
            self.position += 1
            yield value
        self.cycle += 1
        self.position = 0
        self._permutation = None

    def __len__(self) -> int:
        return self.padded_length

    def state_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "position": self.position,
            "permutation": list(self._indices()),
            "seed": self.seed,
            "dataset_length": len(self.data_source),
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if int(state_dict["dataset_length"]) != len(self.data_source) or int(state_dict["batch_size"]) != self.batch_size:
            raise ValueError("sampler state belongs to a different dataset or batch size")
        if int(state_dict["seed"]) != self.seed:
            raise ValueError("sampler seed differs from checkpoint")
        permutation = [int(value) for value in state_dict["permutation"]]
        if len(permutation) != self.padded_length:
            raise ValueError("invalid saved permutation length")
        self.cycle = int(state_dict["cycle"])
        self.position = int(state_dict["position"])
        if not 0 <= self.position <= len(permutation):
            raise ValueError("invalid saved sampler position")
        self._permutation = permutation

    def coverage_stats(self) -> dict[str, int]:
        unique_seen = min(self.position, len(self.data_source))
        padding_seen = max(0, self.position - len(self.data_source))
        return {
            "coverage_cycle": self.cycle,
            "coverage_unique_rollouts": unique_seen,
            "coverage_padding_rollouts": padding_seen,
            "coverage_position": self.position,
        }
