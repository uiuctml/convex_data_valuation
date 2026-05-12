# data/aya_sft/module.py

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from ..base_module import BaseDataModule, FormatterFn, TaskSpec, DataModuleConfig

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Supported languages: language code → Aya "language" full name
# -------------------------------------------------------------------
# 2-letter task code -> Aya display name (used only for logging / metadata)
AYA_LANG_MAP: Dict[str, str] = {
    "ar": "Arabic",
    "bn": "Bengali",
    # "ca": "Catalan",
    "da": "Danish",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "eu": "Basque",
    "fr": "French",
    "gu": "Gujarati",
    "hi": "Hindi",
    "hu": "Hungarian",
    # "hy": "Armenian",
    "id": "Indonesian",
    # "is": "Icelandic",
    "it": "Italian",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    # "nb": "Norwegian Bokmål",
    # "ne": "Nepali",
    "nl": "Dutch",
    "pt": "Portuguese",
    # "ro": "Romanian",
    "ru": "Russian",
    # "sk": "Slovak",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Simplified Chinese",
}

# 2-letter task code -> 3-letter code used by dolly_machine_translated "language" column
AYA_ISO3_MAP: Dict[str, str] = {
    "ar": "arb",
    "bn": "ben",
    # "ca": "cat",
    "da": "dan",
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "eu": "eus",
    "fr": "fra",
    "gu": "guj",
    "hi": "hin",
    "hu": "hun",
    # "hy": "hye",
    "id": "ind",
    # "is": "isl",
    "it": "ita",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    # "nb": "nob",
    # "ne": "nep",
    "nl": "nld",
    "pt": "por",
    # "ro": "ron",
    "ru": "rus",
    # "sk": "slk",
    "sr": "srp",
    "sv": "swe",
    "ta": "tam",
    "te": "tel",
    "uk": "ukr",
    "vi": "vie",
    "zh": "zho",
}


def _canonical_lang(task_name: str) -> Optional[str]:
    """Task name is just a language code; normalize and validate."""
    t = task_name.strip().lower()
    if t in AYA_LANG_MAP:
        return t
    return None


class AyaSFTDataModule(BaseDataModule):
    """
    Data module for Aya SFT dataset.

    Task name = language code (e.g., "en", "de", "fr", "zh").
    We map this code to Aya's `language` column and filter the dataset by it.
    """

    def __init__(
        self,
        tasks: List[TaskSpec],
        cfg: DataModuleConfig,
        formatter: FormatterFn,
        collator,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        super().__init__(tasks, cfg, formatter, collator, tokenizer)
        self._full_dataset: Optional[DatasetDict] = None

    # -------- canonical name --------

    def _canonical_task_name(self, task_name: str) -> str:
        code = _canonical_lang(task_name)
        return code if code is not None else task_name

    # -------- loading --------

    def _load_task(self, task_name: str) -> DatasetDict:
        lang_code = self._canonical_task_name(task_name)

        if self._full_dataset is None:
            self._full_dataset = self._load_full_dataset()

        # map code → Aya "language" field
        aya_name = AYA_LANG_MAP.get(lang_code)
        if aya_name is None:
            logger.warning(f"[AyaSFT] Unknown language '{task_name}', returning full dataset")
            return self._full_dataset

        return self._filter_by_language(aya_name, AYA_ISO3_MAP.get(lang_code))

    def _load_full_dataset(self) -> DatasetDict:
        print("Loading Aya dataset from HuggingFace (CohereLabs/aya_dataset)")
        aya = load_dataset("CohereLabs/aya_dataset")
        print(f"Aya dataset splits: {list(aya.keys())}")

        print("Loading Aya evaluation suite: dolly_machine_translated (test split)")
        eval_ds = load_dataset(
            "CohereLabs/aya_evaluation_suite",
            "dolly_machine_translated",
            split="test",
        )

        # Replace / add the test split in the main dataset dict
        # (If aya already has 'test', we overwrite it.)
        aya = DatasetDict(dict(aya))
        aya["test"] = eval_ds

        print(f"After override, Aya splits: {list(aya.keys())}")
        return aya

    def _filter_by_language(self, aya_language_name: str, iso3: str) -> DatasetDict:
        """
        Filter:
        - aya_dataset splits: by `language_code` (2-letter)
        - eval-suite test split: by `language` (3-letter)   [dolly_machine_translated]
        """

        def filter_fn(example):
            # Prefer language_code when present (Aya train/val)
            lang2 = (example.get("language_code") or "").strip().lower()
            if lang2:
                return lang2 == iso3

            # Fallback: some splits/datasets may only have `language`:
            # - Aya: full language name (e.g., "English")
            # - Eval suite: iso3 code (e.g., "eng")
            lang = (example.get("language") or "").strip()
            if lang == aya_language_name:
                return True
            if iso3 is not None and lang.lower() == iso3:
                return True
            return False

        out = {}
        for split, ds in self._full_dataset.items():
            ds_f = ds.filter(filter_fn, desc=f"filter_lang_{aya_language_name}_{split}")
            print(f"[AyaSFT] {aya_language_name} ({iso3}) {split}: {len(ds_f)} examples")
            out[split] = ds_f

        return DatasetDict(out)

    # -------- k-shot sampling --------

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

    # -------- formatter override --------

    def _apply_formatter(self, ds: DatasetDict, task: str) -> DatasetDict:
        out = {}
        for split, d in ds.items():
            print(f"[AyaSFT/format] {task} ({split}) → applying formatter ({len(d)} rows)")

            formatted = d.map(
                lambda ex, idx: {**self.formatter(ex, task, tokenizer=self.tokenizer), "id": idx},
                with_indices=True,
                remove_columns=d.column_names,
                desc=f"format:{task}:{split}",
                load_from_cache_file=True,
            )

            # Filter rows where formatter returned None
            if "prompt" in formatted.column_names:
                filtered = formatted.filter(
                    lambda ex: ex.get("prompt") is not None,
                    desc=f"filter_none:{task}:{split}",
                )
            else:
                filtered = formatted

            print(
                f"[AyaSFT/format] {task}/{split}: kept {len(filtered)}/{len(d)}, "
                f"skipped {len(d) - len(filtered)}"
            )
            out[split] = filtered

        return DatasetDict(out)

    # -------- kshot cache base directory --------

    def _kshot_dir_root(self) -> str:
        return "aya_sft_kshot"