from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import PreTrainedTokenizerBase

from ..base_module import BaseDataModule, FormatterFn, TaskSpec, DataModuleConfig

logger = logging.getLogger(__name__)


# Task name → (HuggingFace ID, split, format)
# format: "messages" = has `messages` field, "instruction" = has `instruction`/`response` fields
TULU_PERSONAS_DATASETS: Dict[str, Dict] = {
    "instruction-following": {"hf_id": "allenai/tulu-3-sft-personas-instruction-following", "split": "train", "format": "messages"},
    "math-filtered": {"hf_id": "allenai/tulu-3-sft-personas-math-filtered", "split": "train", "format": "messages"},
    "math-grade-filtered": {"hf_id": "allenai/tulu-3-sft-personas-math-grade-filtered", "split": "train", "format": "messages"},
    "algebra": {"hf_id": "allenai/tulu-3-sft-personas-algebra", "split": "train", "format": "messages"},
    "code": {"hf_id": "allenai/tulu-3-sft-personas-code", "split": "train", "format": "messages"},
    "evol-instruct": {"hf_id": "SurgeGlobal/Evol-Instruct", "split": "train", "format": "instruction"},
    "smol-smoltalk": {"hf_id": "HuggingFaceTB/smol-smoltalk", "split": "train", "format": "messages"},
    "openhermes": {"hf_id": "HuggingFaceTB/OpenHermes-2.5-H4", "split": "train_sft", "format": "messages"},
}

# Split suffixes: task names ending with -A or -B select first/second half
SPLIT_SUFFIXES = ("-A", "-B")


class TuluPersonasSFTDataModule(BaseDataModule):
    """
    Data module for Tulu-3 SFT Personas datasets.

    Task name = subset name (e.g., "instruction-following", "math-filtered").
    Each task loads from its own HF dataset, samples `max_samples` examples,
    and splits into train/validation.
    """

    def __init__(
        self,
        tasks: List[TaskSpec],
        cfg: DataModuleConfig,
        formatter: FormatterFn,
        collator,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_samples: int = 3000,
        sample_seed: int = 42,
        val_ratio: float = 0.1,
    ):
        super().__init__(tasks, cfg, formatter, collator, tokenizer)
        self.max_samples = max_samples
        self.sample_seed = sample_seed
        self.val_ratio = val_ratio

    def _canonical_task_name(self, task_name: str) -> str:
        return task_name.strip().lower()

    def _load_task(self, task_name: str) -> DatasetDict:
        canon = self._canonical_task_name(task_name)

        # Check for -A/-B split suffix (e.g., "math-filtered-A" → base="math-filtered", half=0)
        half = None
        base_name = canon
        if canon.endswith("-a"):
            base_name = canon[:-2]
            half = 0
        elif canon.endswith("-b"):
            base_name = canon[:-2]
            half = 1

        ds_info = TULU_PERSONAS_DATASETS.get(base_name)
        if ds_info is None:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Supported: {list(TULU_PERSONAS_DATASETS.keys())} (with optional -A/-B suffix)"
            )

        hf_id = ds_info["hf_id"]
        hf_split = ds_info["split"]
        fmt = ds_info["format"]

        # Use cached full sample if available (avoids re-downloading for A/B splits)
        cache_key = f"{base_name}_seed{self.sample_seed}_n{self.max_samples}"
        if not hasattr(self, "_dataset_cache"):
            self._dataset_cache: Dict[str, Dataset] = {}

        if cache_key not in self._dataset_cache:
            print(f"[TuluPersonas] Loading {hf_id} (split={hf_split}) from HuggingFace...")
            ds = load_dataset(hf_id, split=hf_split)
            print(f"[TuluPersonas] {hf_id}: {len(ds)} total examples")

            # Normalize format: convert instruction/response to messages format
            if fmt == "instruction":
                print(f"[TuluPersonas] Converting instruction/response → messages format")
                ds = ds.map(
                    lambda ex: {
                        "messages": [
                            {"role": "user", "content": ex["instruction"]},
                            {"role": "assistant", "content": ex["response"]},
                        ]
                    },
                    desc=f"convert:{base_name}",
                )

            # Sample max_samples
            n = len(ds)
            k = min(self.max_samples, n)
            rng = np.random.default_rng(self.sample_seed)
            indices = sorted(rng.choice(n, size=k, replace=False).tolist())
            ds = ds.select(indices)
            print(f"[TuluPersonas] Sampled {k} examples (seed={self.sample_seed})")
            self._dataset_cache[cache_key] = ds
        else:
            ds = self._dataset_cache[cache_key]
            print(f"[TuluPersonas] Using cached {base_name} ({len(ds)} examples)")

        # Apply A/B split if requested
        if half is not None:
            mid = len(ds) // 2
            if half == 0:
                ds = ds.select(range(mid))
            else:
                ds = ds.select(range(mid, len(ds)))
            print(f"[TuluPersonas] Half {'A' if half == 0 else 'B'}: {len(ds)} examples")

        # Split into train / validation
        n_val = max(1, int(len(ds) * self.val_ratio))
        n_train = len(ds) - n_val

        train_ds = ds.select(range(n_train))
        val_ds = ds.select(range(n_train, len(ds)))

        print(f"[TuluPersonas] {canon}: train={len(train_ds)}, val={len(val_ds)}")

        return DatasetDict({
            "train": train_ds,
            "validation": val_ds,
        })

    def _apply_kshot(self, ds: DatasetDict, k: int, seed: int, split: str) -> DatasetDict:
        if split not in ds:
            raise ValueError(f"k-shot split '{split}' not found (in {list(ds.keys())})")

        base = ds[split]
        n = len(base)
        if n <= k:
            return ds

        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(n, size=k, replace=False).tolist())

        out = dict(ds)
        out["train"] = base.select(idx)
        return DatasetDict(out)

    def _apply_formatter(self, ds: DatasetDict, task: str) -> DatasetDict:
        out = {}
        for split, d in ds.items():
            print(f"[TuluPersonas/format] {task} ({split}) → applying formatter ({len(d)} rows)")

            formatted = d.map(
                lambda ex, idx: {**self.formatter(ex, task, tokenizer=self.tokenizer), "id": idx},
                with_indices=True,
                remove_columns=d.column_names,
                desc=f"format:{task}:{split}",
                load_from_cache_file=True,
            )

            # Filter rows where formatter returned None (e.g. empty messages)
            if "input_ids" in formatted.column_names:
                filtered = formatted.filter(
                    lambda ex: ex.get("input_ids") is not None,
                    desc=f"filter_none:{task}:{split}",
                )
            else:
                filtered = formatted

            print(
                f"[TuluPersonas/format] {task}/{split}: kept {len(filtered)}/{len(d)}, "
                f"skipped {len(d) - len(filtered)}"
            )
            out[split] = filtered

        return DatasetDict(out)

    def _kshot_dir_root(self) -> str:
        return "tulu_personas_kshot"
