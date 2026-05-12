import os
import asyncio
from typing import Optional, List, Dict, Any, Tuple
import hashlib
import random
from pathlib import Path

from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
from googletrans import Translator
import json

# import these from your helper file:
# from your_translate_helpers import translate_text_async, CONCURRENCY, LANGS
# (I’ll assume you already have LANGS list like in LIMR)

CONCURRENCY = 48
MAX_CHARS = 4500
MAX_RETRIES = 6
BASE_SLEEP = 0.8

LANGS = [
    ("en", "English"),
    ("zh", "Chinese"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("it", "Italian"),
    ("nl", "Dutch"),
    ("ru", "Russian"),
    ("cs", "Czech"),
    ("pl", "Polish"),
    ("ar", "Arabic"),
    ("fa", "Persian"),
    ("he", "Hebrew"),
    ("tr", "Turkish"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("vi", "Vietnamese"),
    ("th", "Thai"),
    ("id", "Indonesian"),
    ("ms", "Malay"),
    ("lo", "Lao"),
    ("my", "Burmese"),
    ("ceb", "Cebuano"),
    ("km", "Khmer"),
    ("tl", "Tagalog"),
    ("hi", "Hindi"),
    ("bn", "Bengali"),
    ("ur", "Urdu"),
]

# ---- paste your translate_text_async here (from my previous message) ----
# translate_text_async(translator, text, dest_lang, src_lang="auto") -> str

CACHE_PATH = Path(os.environ.get("DATA_DIR", ".")) / "translation_cache_gsm8k.jsonl"

def load_cache() -> Dict[Tuple[str, str], str]:
    cache: Dict[Tuple[str, str], str] = {}
    if not CACHE_PATH.exists():
        return cache
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                cache[(obj["lang"], obj["hash"])] = obj["text"]
            except Exception:
                continue
    print(f"[cache] loaded {len(cache)}")
    return cache

_CACHE = load_cache()

def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def split_by_paragraphs(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    paras = text.split("\n\n")
    chunks, cur = [], ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) > max_chars:
            # hard split
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i+max_chars])
            continue
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= max_chars:
            cur = cur + "\n\n" + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

def cleanup_quotes(text: str) -> str:
    # remove a single unmatched trailing quote
    t = (text or "").strip()
    if t.endswith('"') and t.count('"') == 1:
        t = t[:-1].rstrip()
    return t

async def translate_text_async(
    translator: Translator,
    text: str,
    dest_lang: str,
    src_lang: Optional[str] = "auto",
) -> str:
    if not text or not text.strip():
        return text

    # googletrans uses zh-cn
    if dest_lang == "zh":
        dest_lang = "zh-cn"

    # don’t translate English
    if dest_lang in {"en", "en-us", "en-gb"}:
        return cleanup_quotes(text)

    key = (dest_lang, _h(text))
    if key in _CACHE:
        return cleanup_quotes(_CACHE[key])

    chunks = split_by_paragraphs(text, MAX_CHARS)
    outs: List[str] = []

    for chunk in chunks:
        translated = None
        for attempt in range(MAX_RETRIES):
            try:
                res = await translator.translate(chunk, dest=dest_lang, src=src_lang)
                translated = res.text
                break
            except Exception:
                await asyncio.sleep(BASE_SLEEP * (2 ** attempt) + random.uniform(0, 0.3))
        outs.append(translated if translated is not None else chunk)

    out = cleanup_quotes("\n\n".join(outs))
    _CACHE[key] = out

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"lang": key[0], "hash": key[1], "text": out}, ensure_ascii=False) + "\n")

    return out

async def build_dataset_async(
    out_repo: str,
    hf_token: Optional[str] = None,
    src_dataset: str = "open-r1/Big-Math-RL-Verified-Processed",
    subset: str = "all",
    split: str = "train",
    keep_source_value: str = "gsm8k",
    train_first_n: Optional[int] = None,
):
    if hf_token:
        login(token=hf_token)

    ds = load_dataset(src_dataset, subset, split=split)

    # keep only GSM8K
    if "source" not in ds.column_names:
        raise ValueError(f"'source' not in columns: {ds.column_names}")
    ds = ds.filter(lambda x: x["source"] == keep_source_value)
    print(f"[load] {keep_source_value} rows: {len(ds)}")

    if "prompt" not in ds.column_names or "solution" not in ds.column_names:
        raise ValueError(f"Need prompt+solution. Columns: {ds.column_names}")

    prompts: List[str] = ds["prompt"]
    solutions: List[str] = ds["solution"]  # KEEP AS-IS

    rows: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(CONCURRENCY)

    async with Translator() as translator:
        for lang_code, lang_name in LANGS:
            print(f"\n[translate] prompt -> {lang_code}")

            async def tr_prompt(p: str) -> str:
                async with sem:
                    return await translate_text_async(translator, p, lang_code)

            t_prompts = await asyncio.gather(*[tr_prompt(p) for p in prompts])

            for i in range(len(ds)):
                r = dict(ds[i])
                r["prompt"] = t_prompts[i]
                r["solution"] = solutions[i]   # unchanged (prevents $...$ / \\text{...} corruption)
                r["lang"] = lang_code
                r["language"] = lang_name
                r["original_index"] = i
                rows.append(r)

    out_ds = Dataset.from_list(rows)

    if train_first_n is not None:
        train_ds = out_ds.filter(lambda x: x["original_index"] < train_first_n)
        val_ds   = out_ds.filter(lambda x: x["original_index"] >= train_first_n)
        train_ds = train_ds.remove_columns(["original_index"])
        val_ds   = val_ds.remove_columns(["original_index"])
        dsd = DatasetDict({"train": train_ds, "validation": val_ds})
    else:
        dsd = DatasetDict({"train": out_ds.remove_columns(["original_index"])})

    dsd.push_to_hub(out_repo)
    print(f"\n✅ pushed: {out_repo}")


def main():
    out_repo = "cindy2000sh/Big-Math-RL-GSM8K-googletrans"
    hf_token = os.environ.get("HF_TOKEN", None)

    asyncio.run(
        build_dataset_async(
            out_repo=out_repo,
            hf_token=hf_token,
            train_first_n=None,  # or 1000 like LIMR
        )
    )


if __name__ == "__main__":
    main()