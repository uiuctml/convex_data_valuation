import copy
import yaml
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, asdict

class DotDict(dict):
    """allow dot notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __deepcopy__(self, memo):
        """Support deepcopy by creating a new DotDict with deepcopied contents"""
        return DotDict(copy.deepcopy(dict(self), memo))
    
    def __reduce__(self):
        """Support pickling/unpickling"""
        return (DotDict, (dict(self),))
    
    # ---- OmegaConf-like API ----
    def to_container(self):
        """
        Convert to plain Python containers (dict/list/tuple), recursively.
        `resolve` is kept for API parity; ignored here.
        """
        return self._to_plain(self, memo={})

    @classmethod
    def _to_plain(cls, obj, memo):
        oid = id(obj)
        if oid in memo:
            return memo[oid]

        # 1) Handle DotDict and any Mapping first (prevents recursion via .to_container)
        if isinstance(obj, Mapping):
            out = {}
            memo[oid] = out
            for k, v in obj.items():
                out[k] = cls._to_plain(v, memo)
            return out

        # 2) Sequences (but not str/bytes)
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            out = []
            memo[oid] = out
            out.extend(cls._to_plain(v, memo) for v in obj)
            return out

        # 3) Dataclasses
        if is_dataclass(obj):
            d = asdict(obj)
            memo[oid] = d  # asdict returns a fresh dict; still memoize to be safe
            return cls._to_plain(d, memo)

        # 4) Foreign objects that expose .to_container (NOT DotDict)
        if hasattr(obj, "to_container") and callable(getattr(obj, "to_container")) and not isinstance(obj, DotDict):
            try:
                return obj.to_container(resolve=True)
            except TypeError:
                return obj.to_container()

        # 5) Plain objects with __dict__
        try:
            attrs = vars(obj)
        except TypeError:
            # 6) Primitives / everything else
            return obj
        else:
            out = {}
            memo[oid] = out
            for k, v in attrs.items():
                out[k] = cls._to_plain(v, memo)
            return out
   

def _dedup_tasks_preserve_case(tasks):
    """
    Deduplicate task names case-insensitively, preserving order and original casing.
    Example:
        ["SST-2", "sst-2", "MNLI"] -> ["SST-2", "MNLI"]
    """
    if tasks is None:
        return None

    seen_lower = set()
    deduped = []
    for t in tasks:
        key = t.strip().lower()
        if key not in seen_lower:
            seen_lower.add(key)
            deduped.append(t.strip())
    return deduped


def load_yaml_config(path: str, base_path: str = None):
    """Load a YAML config file, optionally merging with a base config."""
    with open(path, "r") as f:
        user = yaml.safe_load(f)

    if base_path is not None:
        with open(base_path, "r") as f:
            base = yaml.safe_load(f)
        base.update(user)
        cfg = DotDict(base)
    else:
        cfg = DotDict(user)

    # Canonicalize task list: dedup but preserve original names
    if "tasks" in cfg and cfg.tasks is not None:
        cfg.tasks = _dedup_tasks_preserve_case(cfg.tasks)

    # Ensure target_task matches one of the tasks (case-insensitive match)
    if "target_task" in cfg and cfg.target_task is not None and "tasks" in cfg:
        for t in cfg.tasks:
            if t.lower() == cfg.target_task.strip().lower():
                cfg.target_task = t
                break

    return cfg