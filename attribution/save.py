from __future__ import annotations
"""
ResultStore: a small utility to persist attribution scores + method-specific artifacts.

- Keeps base class minimal (no changes to DataAttribution).
- Children (e.g., OneStepGradientAttribution, TaskVectorAttribution) call this to save:
    * scores (Dict[str, float]) -> JSON
    * artifacts (tensors, state_dicts, numpy arrays, lists/dicts of tensors, DataFrames, etc.)
    * run metadata (method name, target/aux tasks, model info, custom notes)
"""

import json
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch
import pandas as pd


# ----------------------------- Dataclasses -----------------------------

@dataclass
class AttributionRunInfo:
    method_name: str
    target_task: str
    aux_tasks: Iterable[str] = field(default_factory=list)
    model_name: Optional[str] = None
    device: Optional[str] = None
    timestamp: Optional[str] = None  # auto-filled if None
    extra: Dict[str, Any] = field(default_factory=dict)


# ----------------------------- Utilities ------------------------------

def _now_stamp() -> str:
    # e.g., 2025-10-23_14-37-05
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def _slugify(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9_.-]", "", s)
    return s or "run"

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _json_dump(obj: Any, path: Path) -> None:
    def default(o):
        if torch is not None and isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set,)):
            return list(o)
        # dataclasses etc.
        try:
            return asdict(o)  # type: ignore
        except Exception:
            pass
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=default)

def _save_tensor(obj: "torch.Tensor", path: Path) -> None:  # type: ignore[name-defined]
    path = path.with_suffix(".pt")
    torch.save(obj.detach().cpu(), path)

def _save_state_dict(sd: Dict[str, "torch.Tensor"], path: Path) -> None:  # type: ignore[name-defined]
    path = path.with_suffix(".pt")
    torch.save({k: v.detach().cpu() for k, v in sd.items()}, path)

def _save_state_dict_collection(obj: Dict[str, Dict[str, "torch.Tensor"]], path: Path) -> None:  # type: ignore[name-defined]
    """Save a dictionary of state_dicts as a single .pt file."""
    path = path.with_suffix(".pt")
    result = {}
    for key, state_dict in obj.items():
        result[key] = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(result, path)

def _save_numpy(arr: np.ndarray, path: Path) -> None:
    path = path.with_suffix(".npy")
    np.save(path, arr)

def _save_dataframe(df: "pd.DataFrame", path: Path) -> None:  # type: ignore[name-defined]
    # Prefer parquet if available, else CSV
    try:
        path = path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    except Exception:
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)

def _save_text(s: str, path: Path) -> None:
    path = path.with_suffix(".txt")
    with path.open("w", encoding="utf-8") as f:
        f.write(s)

def _save_jsonable(obj: Any, path: Path) -> None:
    path = path.with_suffix(".json")
    _json_dump(obj, path)

def _save_bytes(b: bytes, path: Path) -> None:
    with path.open("wb") as f:
        f.write(b)


# ----------------------------- ResultStore -----------------------------

class ResultStore:
    """
    Filesystem-backed store for attribution runs.

    Typical usage (inside an attribution method):

        store = ResultStore(root_dir="outputs/attribution")
        run_dir = store.new_run_dir(method_name="one_step", target_task="SST-2")
        store.save_run_info(run_dir, AttributionRunInfo(...))
        store.save_scores(run_dir, scores)
        store.save_artifact(run_dir, "target_avg_grad", target_grad_tensor)
        for name, g in aux_grads.items():
            store.save_artifact(run_dir, f"aux_{name}_grad", g)

    You can also call `store.write_run(...)` to do it in one shot.
    """

    def __init__(self, root_dir: Union[str, Path] = "outputs/attribution") -> None:
        self.root_dir = _ensure_dir(root_dir)

    # ---- Run directory management ----
    def new_run_dir(self, *, method_name: str, target_task: str) -> Path:
        stamp = _now_stamp()
        base = f"{_slugify(method_name)}_{_slugify(target_task)}_{stamp}"
        run_dir = _ensure_dir(self.root_dir / base)
        _ensure_dir(run_dir / "artifacts")
        return run_dir

    # ---- Save primitives ----
    def save_run_info(self, run_dir: Union[str, Path], info: AttributionRunInfo) -> Path:
        run_dir = Path(run_dir)
        if not info.timestamp:
            info.timestamp = _now_stamp()
        p = run_dir / "run_info.json"
        _json_dump(asdict(info), p)
        return p

    def save_scores(self, run_dir: Union[str, Path], scores: Dict[str, float]) -> Path:
        run_dir = Path(run_dir)
        p = run_dir / "scores.json"
        _json_dump(scores, p)
        return p

    def save_artifact(self, run_dir: Union[str, Path], name: str, obj: Any) -> Path:
        """Save a single artifact under run_dir/artifacts/<name>.<ext>."""
        art_dir = _ensure_dir(Path(run_dir) / "artifacts")
        base = art_dir / _slugify(name)

        # torch Tensor or state_dict
        if torch is not None:
            if isinstance(obj, torch.Tensor):
                _save_tensor(obj, base)
                return base.with_suffix(".pt")
            if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
                _save_state_dict(obj, base)
                return base.with_suffix(".pt")

        # numpy array
        if isinstance(obj, np.ndarray):
            _save_numpy(obj, base)
            return base.with_suffix(".npy")

        # pandas DataFrame
        if pd is not None and isinstance(obj, pd.DataFrame):
            _save_dataframe(obj, base)
            # suffix determined inside
            return base.with_suffix(".parquet") if (base.with_suffix(".parquet")).exists() else base.with_suffix(".csv")

        # bytes
        if isinstance(obj, (bytes, bytearray)):
            _save_bytes(bytes(obj), base.with_suffix(".bin"))
            return base.with_suffix(".bin")

        # str
        if isinstance(obj, str):
            _save_text(obj, base)
            return base.with_suffix(".txt")

        # list/dict of tensors or jsonable data
        if isinstance(obj, (list, tuple)):
            # Try to detect list/tuple of tensors
            if torch is not None and all(isinstance(x, torch.Tensor) for x in obj):
                # save as a single .pt list
                _save_tensor(torch.stack([x.detach().cpu() for x in obj]), base)
                return base.with_suffix(".pt")
            # else try json
            _save_jsonable(obj, base)
            return base.with_suffix(".json")

        if isinstance(obj, dict):
            # Check if it's a dict of state_dicts (e.g., aux_avg_grads)
            # Save as a single .pt file containing the entire dictionary
            if torch is not None and all(
                isinstance(v, dict) and all(torch.is_tensor(t) for t in v.values())
                for v in obj.values()
            ):
                # Dict of state_dicts - save as single .pt file
                _save_state_dict_collection(obj, base)
                return base.with_suffix(".pt")
            
            # If mixed content, try JSON; if not serializable, fallback to per-item files
            try:
                _save_jsonable(obj, base)
                return base.with_suffix(".json")
            except Exception:
                # fallback: save each item individually
                for k, v in obj.items():
                    self.save_artifact(art_dir, f"{name}_{k}", v)
                return art_dir

        # last resort: JSON stringify
        _save_jsonable(obj, base)
        return base.with_suffix(".json")

    # ---- One-shot convenience ----
    def write_run(
        self,
        *,
        method_name: str,
        target_task: str,
        scores: Dict[str, float],
        artifacts: Optional[Dict[str, Any]] = None,
        run_info: Optional[AttributionRunInfo] = None,
    ) -> Path:
        """
        Create a new run dir, write run_info/scores, and dump artifacts.

        Returns the path to the created run directory.
        """
        run_dir = self.new_run_dir(method_name=method_name, target_task=target_task)
        if run_info is None:
            run_info = AttributionRunInfo(
                method_name=method_name,
                target_task=target_task,
                aux_tasks=list(scores.keys()),
            )
        self.save_run_info(run_dir, run_info)
        self.save_scores(run_dir, scores)

        if artifacts:
            for name, obj in artifacts.items():
                self.save_artifact(run_dir, name, obj)

        return run_dir