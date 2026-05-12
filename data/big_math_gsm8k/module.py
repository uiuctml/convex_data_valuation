# data/multilingual_math/big_math_rl_gsm8k_googletrans_module.py

from __future__ import annotations
import logging
from typing import List, Optional, Set

from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from ..base_module import BaseDataModule, FormatterFn, TaskSpec, DataModuleConfig

logger = logging.getLogger(__name__)


# Same language set as LIMR/googletrans script
BIG_MATH_RL_LANGUAGES: Set[str] = {
    "en", "zh",
    "de", "fr", "es", "pt", "it", "nl",
    "ru", "cs", "pl",
    "ar", "fa", "he", "tr",
    "ja", "ko",
    "vi", "th", "id", "ms", "lo", "my", "ceb", "km", "tl",
    "hi", "bn", "ur",
}


def _canonical_lang(name: str) -> Optional[str]:
    n = (name or "").strip().lower()
    if n in {"zh-cn", "zh-hans"}:
        n = "zh"
    if n in BIG_MATH_RL_LANGUAGES:
        return n
    return None


class BigMathRLGSM8KGoogleTransRLDataModule(BaseDataModule):
    """
    Data module for RL/GRPO training on the Big-Math-RL GSM8K slice after googletrans multilingual expansion.

    Differences vs LIMR module:
      - Dataset uses `solution` as supervision; formatter maps solution -> answer.
      - No real validation/test: create dummy empty splits to satisfy downstream code paths.
      - Format ALL datapoints ONCE, then filter by language per task.
    """

    def __init__(
        self,
        tasks: List[TaskSpec],
        cfg: DataModuleConfig,
        formatter: FormatterFn,
        collator,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dataset_name: str = "cindy2000sh/Big-Math-RL-GSM8K-googletrans",
        drop_missing_answer: bool = True,
    ):
        super().__init__(tasks, cfg, formatter, collator, tokenizer)
        self.dataset_name = dataset_name
        self.drop_missing_answer = drop_missing_answer

        self._full_dataset: Optional[DatasetDict] = None
        self._formatted_full_dataset: Optional[DatasetDict] = None  # formatted once

    def prepare(self):
        """
        Our _load_task already returns a formatted DatasetDict (prompt/answer/...).
        So we bypass BaseDataModule._apply_formatter to avoid formatting twice.
        
        But we MUST still apply k-shot if specified in the TaskSpec.
        """
        for t in self.tasks:
            task_name = t.name
            ds = self._load_task(task_name)  # already formatted + filtered + language-sliced

            # Apply k-shot if specified (important for attribution methods!)
            if t.k_shot is not None:
                canon = self._canonical_task_name(task_name)
                tag = f"{canon}-k{t.k_shot}-seed{t.k_shot_seed}"
                t.tag = tag
                
                logger.info(f"[BigMathRL] Applying k-shot: task={task_name}, k={t.k_shot}, seed={t.k_shot_seed}")

                if (
                    self.cfg.save_kshot_locally
                    and self.cfg.datadir
                    and self._kshot_exists(task_name, t.k_shot, t.k_shot_seed)
                ):
                    ds = self._load_kshot(task_name, t.k_shot, t.k_shot_seed)
                else:
                    ds = self._apply_kshot(
                        ds,
                        k=t.k_shot,
                        seed=t.k_shot_seed,
                        split=t.k_shot_split,
                    )
                    if self.cfg.save_kshot_locally and self.cfg.datadir:
                        self._save_kshot(
                            task_name,
                            ds,
                            k=t.k_shot,
                            seed=t.k_shot_seed,
                        )
            else:
                t.tag = self._canonical_task_name(task_name)

            self.raw_ds[task_name] = ds
            self.prompted_ds[task_name] = ds  # IMPORTANT: already formatted

    # -------- canonical name --------

    def _canonical_task_name(self, task_name: str) -> str:
        lang = _canonical_lang(task_name)
        return lang if lang is not None else task_name

    # -------- loading --------

    def _load_task(self, task_name: str) -> DatasetDict:
        canonical = self._canonical_task_name(task_name)

        if self._full_dataset is None:
            self._full_dataset = self._load_full_dataset()

        # Format everything once (expensive) then reuse.
        if self._formatted_full_dataset is None:
            self._formatted_full_dataset = self._format_full_dataset_once(self._full_dataset)

        if canonical in BIG_MATH_RL_LANGUAGES:
            return self._filter_formatted_by_language(self._formatted_full_dataset, canonical)

        logger.warning(f"Unknown task name '{task_name}', returning formatted full dataset")
        return self._formatted_full_dataset

    def _load_full_dataset(self) -> DatasetDict:
        """
        Load dataset from HF. If only train exists (typical), we keep it and create
        dummy empty validation/test splits (as requested).
        """
        logger.info(f"Loading Big-Math-RL GSM8K googletrans from HF: {self.dataset_name}")
        ds = load_dataset(self.dataset_name)
        logger.info(f"Loaded splits: {list(ds.keys())}")

        if "train" not in ds:
            raise ValueError(f"Expected a 'train' split. Found: {list(ds.keys())}")

        # Create dummy validation/test if missing
        train = ds["train"]
        out = {"train": train}

        # Empty slices (same schema) for dummy splits
        empty = train.select([])

        if "validation" not in ds:
            logger.info("No validation split found; creating dummy empty validation split")
            out["validation"] = empty
        else:
            out["validation"] = ds["validation"]

        if "test" not in ds:
            logger.info("No test split found; creating dummy empty test split")
            out["test"] = empty
        else:
            out["test"] = ds["test"]

        return DatasetDict(out)

    # -------- one-time formatting --------

    def _format_full_dataset_once(self, ds: DatasetDict) -> DatasetDict:
        """
        Apply formatter ONCE to each split, producing standardized schema:
          prompt/answer/problem/lang/task/etc.
        Then reuse for all languages.
        """
        out = {}
        for split, d in ds.items():
            logger.info(f"[BigMathRL format-once] split={split} n={len(d)}")

            # If dummy empty splits, just propagate an empty dataset with the formatted schema.
            if len(d) == 0:
                # Create an empty formatted dataset by mapping on empty (keeps columns)
                # but datasets may drop schema on empty; easiest is to select([]) after formatting train once.
                # We'll handle this after formatting train: for now, placeholder.
                out[split] = d
                continue

            def safe_formatter(ex):
                # Formatter should read ex["prompt"] and ex["solution"], and output ex["answer"] accordingly.
                res = self.formatter(ex, task="__all__", tokenizer=self.tokenizer)
                if res is None:
                    # Return Nones and filter later; do NOT crash here.
                    return {
                        "prompt": None,
                        "answer": None,
                        "problem": None,
                        "lang": ex.get("lang"),
                        "task": "__all__",
                    }
                return res

            formatted = d.map(
                safe_formatter,
                remove_columns=d.column_names,
                desc=f"format_once:{split}",
                load_from_cache_file=False,
                new_fingerprint=f"big_math_rl_gsm8k_fmt_v3_{split}",
            )

            def keep_fn(ex):
                if ex.get("prompt") is None:
                    return False
                if self.drop_missing_answer and (ex.get("answer") is None or str(ex.get("answer")).strip() == ""):
                    return False
                return True

            filtered = formatted.filter(keep_fn, desc=f"filter_once:{split}")
            logger.info(f"[BigMathRL format-once] {split}: kept {len(filtered)}/{len(d)}")
            out[split] = filtered

        # Ensure dummy splits exist with the *formatted* schema:
        # If validation/test were empty, we want them empty but with the same columns as train.
        if "train" in out and len(out["train"]) > 0:
            train_formatted = out["train"]
            for split in ("validation", "test"):
                if split in out and len(out[split]) == 0:
                    out[split] = train_formatted.select([])

        return DatasetDict(out)

    # -------- filtering formatted ds by language (cheap) --------

    def _filter_formatted_by_language(self, ds: DatasetDict, lang: str) -> DatasetDict:
        logger.info(f"[BigMathRL filter] lang={lang}")

        def norm_lang(x: str) -> str:
            x = (x or "").lower()
            if x in {"zh-cn", "zh-hans"}:
                return "zh"
            return x

        def filter_fn(ex):
            return norm_lang(ex.get("lang")) == lang

        out = {}
        for split, d in ds.items():
            if len(d) == 0:
                out[split] = d  # keep empty
                continue
            out[split] = d.filter(filter_fn, desc=f"filter_lang_{lang}_{split}")
            logger.info(f"[BigMathRL filter] {lang} {split}: {len(out[split])}")

        return DatasetDict(out)

    # -------- k-shot (uniform) --------

    def _apply_kshot(
        self,
        ds: DatasetDict,
        k: int,
        seed: int,
        split: str,
    ) -> DatasetDict:
        if split not in ds:
            raise ValueError(f"k-shot split '{split}' not found. Available: {list(ds.keys())}")

        base = ds[split]
        n = len(base)
        if n == 0 or n <= k:
            return ds

        import numpy as np
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(n, size=k, replace=False).tolist())
        kshot = base.select(idx)

        out = dict(ds)
        out["train"] = kshot
        return DatasetDict(out)

    # -------- k-shot dir root override --------

    def _kshot_dir_root(self) -> str:
        return "big_math_rl_gsm8k_googletrans_kshot"