from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

from datasets import (
    load_from_disk,
    DatasetDict,
)
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


# ------------------------- Task spec & defaults -------------------------
@dataclass
class TaskSpec:
    name: str
    # Optional few-shot config
    k_shot: Optional[int] = None
    k_shot_seed: int = 42
    k_shot_split: str = "train"   # which split to draw k-shot from

    # internal tag (set during load): e.g. "MPQA-k16-seed2025"
    tag: Optional[str] = None

@dataclass
class DataModuleConfig:
    datadir: Optional[str] = None             # root with glue_data/ or other cache dir
    save_kshot_locally: bool = True
    num_workers: int = 4
    train_batch_size: int = 16
    eval_batch_size: int = 64
    pin_memory: bool = True
    drop_last: bool = False

# ------------------------- Formatting hooks -------------------------

# A formatter turns standardized / raw datasets into prompted examples.
# We keep the type hint simple; in practice we often pass tokenizer= as kwarg.
FormatterFn = Callable[[Dict, str], Dict]


# ========================= BASE DATA MODULE =============================

class BaseDataModule(ABC):
    """
    Abstract base class for multitask DataModules.

    Responsibilities (shared):
      - Hold TaskSpec list + DataModuleConfig + collator + optional tokenizer
      - For each task:
          * load raw DatasetDict via _load_task(task_name)
          * optional postprocess/standardize via _postprocess_loaded_ds
          * optional k-shot with caching (_apply_kshot + _save_kshot/_load_kshot)
          * format into prompted_ds via self.formatter
      - Build per-task train/eval DataLoaders in setup_dataloaders()

    Subclasses MUST implement:
      - _load_task(self, task_name) -> DatasetDict
      - _apply_kshot(self, ds, k, seed, split) -> DatasetDict
      - _canonical_task_name(self, task_name) -> str

    Subclasses MAY override:
      - _postprocess_loaded_ds(self, task_name, ds)
      - _pick_train_split(self, task_name, ds_dict)
      - _pick_eval_split(self, task_name, ds_dict)
      - _kshot_dir_root(), _save_kshot(), _load_kshot(), _kshot_exists()
    """

    def __init__(
        self,
        tasks: List[TaskSpec],
        cfg: DataModuleConfig,
        formatter: FormatterFn,
        collator,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.tasks = tasks
        self.cfg = cfg
        self.formatter = formatter
        self.collator = collator
        self.tokenizer = tokenizer

        self.raw_ds: Dict[str, DatasetDict] = {}
        self.prompted_ds: Dict[str, DatasetDict] = {}
        self.train_loaders: Dict[str, DataLoader] = {}
        self.eval_loaders: Dict[str, DataLoader] = {}

    # --------------------- public API ---------------------

    def prepare(self):
        """
        Load, optional postprocess, optional k-shot + cache, then format.
        """
        for t in self.tasks:
            task_name = t.name

            # 1) Load raw
            ds = self._load_task(task_name)

            # 2) Optional standardization/postprocessing
            ds = self._postprocess_loaded_ds(task_name, ds)

            # 3) Optional k-shot with caching
            if t.k_shot is not None:
                canon = self._canonical_task_name(task_name)
                tag = f"{canon}-k{t.k_shot}-seed{t.k_shot_seed}"
                t.tag = tag

                if (
                    self.cfg.save_kshot_locally
                    and self.cfg.datadir
                    and self._kshot_exists(task_name, t.k_shot, t.k_shot_seed)
                ):
                    ds_k = self._load_kshot(task_name, t.k_shot, t.k_shot_seed)
                else:
                    ds_k = self._apply_kshot(
                        ds,
                        k=t.k_shot,
                        seed=t.k_shot_seed,
                        split=t.k_shot_split,
                    )
                    if self.cfg.save_kshot_locally and self.cfg.datadir:
                        self._save_kshot(
                            task_name,
                            ds_k,
                            k=t.k_shot,
                            seed=t.k_shot_seed,
                        )
                ds = ds_k
            else:
                t.tag = self._canonical_task_name(task_name)

            self.raw_ds[task_name] = ds
            self.prompted_ds[task_name] = self._apply_formatter(ds, task_name)

    def setup_dataloaders(self):
        """Create per-task train/eval loaders."""
        # Use train_batch_size from DataModuleConfig (set appropriately by each trainer type)
        train_bs = self.cfg.train_batch_size
        eval_bs = self.cfg.eval_batch_size
        
        for t in self.tasks:
            task_name = t.name
            ds = self.prompted_ds[task_name]
            train_split = self._pick_train_split(task_name, ds)
            eval_split = self._pick_eval_split(task_name, ds)

            self.train_loaders[task_name] = DataLoader(
                ds[train_split],
                batch_size=train_bs,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                drop_last=self.cfg.drop_last,
                collate_fn=self.collator,
            )
            self.eval_loaders[task_name] = DataLoader(
                ds[eval_split],
                batch_size=eval_bs,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                collate_fn=self.collator,
            )

    # ------------------- abstract hooks -------------------

    @abstractmethod
    def _load_task(self, task_name: str) -> DatasetDict:
        """Load a raw DatasetDict for a task name (subclass-specific)."""
        raise NotImplementedError

    @abstractmethod
    def _apply_kshot(
        self,
        ds: DatasetDict,
        k: int,
        seed: int,
        split: str,
    ) -> DatasetDict:
        """Apply dataset-specific k-shot selection."""
        raise NotImplementedError

    @abstractmethod
    def _canonical_task_name(self, task_name: str) -> str:
        """Normalize task names (e.g. 'sst2' → 'SST-2', 'level3' → 'Level_3')."""
        raise NotImplementedError

    # ------------------- overridable helpers -------------------

    def _postprocess_loaded_ds(self, task_name: str, ds: DatasetDict) -> DatasetDict:
        """Optional extra processing on the raw loaded dataset."""
        return ds

    def _pick_train_split(self, task_name: str, ds_dict: DatasetDict) -> str:
        """Default: prefer 'train', otherwise 'validation', otherwise first key."""
        if "train" in ds_dict:
            return "train"
        if "validation" in ds_dict:
            return "validation"
        return sorted(ds_dict.keys())[0]

    def _pick_eval_split(self, task_name: str, ds_dict: DatasetDict) -> str:
        """Default: prefer 'validation', then 'test', then 'train'."""
        if "validation" in ds_dict:
            return "validation"
        if "test" in ds_dict:
            return "test"
        return "train"

    # ------------------- formatter -------------------

    def _apply_formatter(self, ds: DatasetDict, task: str) -> DatasetDict:
        """Apply the formatter to all splits."""
        out = {}
        for split, d in ds.items():
            print(f"[format] {task} ({split}) → applying formatter... ({len(d)} examples)")
            out[split] = d.map(
                # We pass tokenizer as kwarg; formatters that don't use it can ignore.
                # Also add 'id' field with the dataset index for tracking sampled data
                lambda ex, idx: {**self.formatter(ex, task, tokenizer=self.tokenizer), "id": idx},  # type: ignore
                with_indices=True,
                remove_columns=d.column_names,
                desc=f"format:{task}:{split}",
                load_from_cache_file=True,
            )
            print(f"[format] {task} ({split}) → done")
        return DatasetDict(out)

    # ------------------- k-shot caching (generic, HF-native) -------------------

    def _kshot_dir_root(self) -> str:
        """
        Subdirectory under cfg.datadir where k-shot datasets are cached.

        Subclasses can override this to customize structure,
        e.g. 'glue_data/k-shot' vs 'math_kshot'.
        """
        return "k-shot"

    def _kshot_dir(self, task_name: str, k: int, seed: int) -> str:
        root = self.cfg.datadir or ""
        canon = self._canonical_task_name(task_name)
        base = os.path.join(root, self._kshot_dir_root(), canon)
        return os.path.join(base, f"k{k}_seed{seed}")

    def _kshot_exists(self, task_name: str, k: int, seed: int) -> bool:
        path = self._kshot_dir(task_name, k, seed)
        return os.path.isdir(path)

    def _save_kshot(
        self,
        task_name: str,
        ds: DatasetDict,
        k: int,
        seed: int,
    ) -> None:
        """
        Generic cache: DatasetDict.save_to_disk() under a directory.
        """
        path = self._kshot_dir(task_name, k, seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ds.save_to_disk(path)

    def _load_kshot(self, task_name: str, k: int, seed: int) -> DatasetDict:
        path = self._kshot_dir(task_name, k, seed)
        if not os.path.isdir(path):
            raise RuntimeError(f"Expected k-shot cache at {path} for task={task_name}, k={k}, seed={seed}")
        loaded = load_from_disk(path)
        if isinstance(loaded, DatasetDict):
            return loaded
        return DatasetDict({"train": loaded})