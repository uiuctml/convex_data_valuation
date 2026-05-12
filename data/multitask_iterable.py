from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, get_worker_info


class MultiTaskOnTheFlyDataset(Dataset):
    """
    Map-style multitask dataset that samples from multiple task datasets.

    All formatting is done at the formatter level:
      - SFT formatters: pre-tokenize data (return input_ids, attention_mask)
      - RL formatters: apply chat template (return text prompt string)
    
    This dataset just passes through the pre-formatted data and adds task_name.
    """

    def __init__(
        self,
        train_loaders: Dict[str, DataLoader],
        mix: Dict[str, float],
        length: int,
        tokenizer: Any,
        seed: int = 0,
        max_seq_length: Optional[int] = None,
    ):
        self.task_names: List[str] = sorted(train_loaders.keys())
        self.datasets: Dict[str, Any] = {
            t: train_loaders[t].dataset for t in self.task_names
        }

        probs = np.array([mix[t] for t in self.task_names], dtype=float)
        if probs.sum() <= 0:
            raise ValueError(f"Mixture has non-positive total mass: {mix}")
        self.probs = probs / probs.sum()

        self.length = int(length)
        self.sizes = {t: len(ds) for t, ds in self.datasets.items()}
        self.base_seed = int(seed)

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length or getattr(tokenizer, "model_max_length", 2048)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Any:
        w_info = get_worker_info()
        worker_id = w_info.id if w_info is not None else 0
        rank = int(os.environ.get("RANK", 0))

        eff_seed = (
            self.base_seed
            + idx
            + worker_id * 10_000_000
            + rank * 1_000_000_000
        )
        rng = np.random.default_rng(eff_seed)

        # sample task
        task_idx = rng.choice(len(self.task_names), p=self.probs)
        t = self.task_names[task_idx]

        # sample example
        ds = self.datasets[t]
        n = self.sizes[t]
        j = int(rng.integers(0, n))
        ex = ds[j]
        
        # Add task_name if not present
        if isinstance(ex, dict) and "task_name" not in ex:
            ex = dict(ex)
            ex["task_name"] = t

        return ex