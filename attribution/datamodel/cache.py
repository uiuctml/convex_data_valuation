"""
Per-row caching system for expensive datamodel matrix computations.

This module provides a caching mechanism to save and load per-row measurement results,
avoiding expensive recomputation when hyperparameters (sampling config, model config, etc.) 
remain the same.

DESIGN PHILOSOPHY:
  - Cache the EXPENSIVE: per-row model training/evaluation (measure_row calls)
  - Don't cache the CHEAP: linear regression fitting (can refit quickly with different params)

Cache key is based on:
  - row_idx: the row index for deterministic, prefix-invariant caching
  - selection_row: which auxiliaries are included in this row
  - target_task_name
  - mtl_cfg fields: model, training hyperparameters, data processing, etc.
  - aux dataset names and order
  - fit_cfg fields that affect row experiments (include_fraction, seed)
  
EXCLUDED from cache key (cheap operations we may want to re-run):
  LINEAR REGRESSION PARAMETERS (allows re-fitting with different settings):
    - num_rows: excluded for prefix-invariant caching
    - alpha: LASSO regularization strength
    - fit_intercept: intercept fitting option
    - normalize_columns: column normalization option
  
  METRIC AGGREGATION PARAMETERS (only affect coefficient combination AFTER measure_row):
    Note: measure_row() always returns all metrics (e.g., both acc and loss).
    These params only control how we aggregate coefficients across metrics.
    - metric_choice, both_weight
    - metric_orientation, metric_weights
"""

from __future__ import annotations
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from utils.config_utils import DotDict


class RowCache:
    """
    Cache for per-row measurement results in datamodel attribution.
    
    Usage:
        cache = RowCache(cache_dir="outputs/datamodel_cache")
        
        # Try to load cached result
        metrics = cache.load(selection_row, aux_names, mtl_cfg, target_task_name, fit_cfg, row_idx)
        if metrics is None:
            # Cache miss - compute expensive result
            metrics = measure_row(...)
            # Save for future use
            cache.save(selection_row, aux_names, mtl_cfg, target_task_name, metrics, fit_cfg, row_idx)
    """
    
    def __init__(
        self, 
        cache_dir: str | Path = "outputs/datamodel_cache",
        enabled: bool = True,
    ):
        """
        Args:
            cache_dir: Directory to store cached results
            enabled: Whether caching is enabled (can be disabled for debugging)
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _make_hashable(self, obj: Any) -> Any:
        """
        Convert an object to a hashable representation for JSON serialization.
        Handles numpy arrays, lists, dicts, dataclasses, etc.
        """
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [self._make_hashable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._make_hashable(v) for k, v in obj.items()}
        if hasattr(obj, '__dict__'):
            # Handle dataclasses and other objects with __dict__
            return {k: self._make_hashable(v) for k, v in obj.__dict__.items() 
                   if not k.startswith('_')}
        # For other types, convert to string representation
        return str(obj)
    
    def _make_cache_key(
        self,
        selection_row: np.ndarray,
        aux_names: List[str],
        mtl_cfg: DotDict,
        target_task_name: str,
        fit_cfg: Optional[Any] = None,
        row_idx: Optional[int] = None,
    ) -> str:
        """
        Generate a unique cache key based on all relevant parameters.
        
        The key includes:
        - selection_row: which datasets are included (the actual row values)
        - row_idx: the row index (for deterministic RNG-based sampling)
        - aux_names: auxiliary dataset names and their order
        - target_task_name: the target task
        - critical mtl_cfg fields: model, training hyperparameters
        - fit_cfg: datamodel-specific parameters (NOT used for LASSO fitting,
                   but affects which row experiments are run and how)
        """
        # Automatically extract all mtl_cfg fields (converts DotDict to regular dict)
        # This ensures we capture everything without manual listing
        if hasattr(mtl_cfg, '__dict__'):
            config_fields = {k: v for k, v in mtl_cfg.__dict__.items() 
                           if not k.startswith('_')}
        elif isinstance(mtl_cfg, dict):
            config_fields = dict(mtl_cfg)
        else:
            config_fields = {}
        
        # Automatically extract fit_cfg fields, excluding those that don't affect measure_row()
        # We cache the EXPENSIVE measure_row() results (model training/evaluation).
        # We do NOT cache the cheap linear regression fitting.
        # 
        # EXCLUDED from cache key (only affect linear regression or post-processing):
        #   Linear Regression / LASSO parameters:
        #     - num_rows: doesn't affect individual rows (prefix-invariant design)
        #     - alpha: LASSO regularization strength (lambda)
        #     - fit_intercept: whether to fit intercept in LASSO
        #     - normalize_columns: whether to z-score columns before LASSO
        #   Metric aggregation parameters (measure_row() returns ALL metrics; these only
        #   control how we combine them into final coefficients):
        #     - metric_choice: which metric to use ("acc", "loss", "both", "auto")
        #     - both_weight: weight when combining acc and loss
        #     - metric_orientation: which direction is better (+1 or -1 per metric)
        #     - metric_weights: weights for multi-metric aggregation
        EXCLUDED_FIT_PARAMS = {
            # Linear regression parameters (we may want to refit with different settings)
            'num_rows', 'alpha', 'fit_intercept', 'normalize_columns',
            # Metric aggregation (only affects how coefficients are combined, not measurement)
            'metric_choice', 'both_weight', 'metric_orientation', 'metric_weights'
        }
        
        fit_config_fields = {}
        if fit_cfg is not None:
            if hasattr(fit_cfg, '__dict__'):
                # It's a dataclass - extract all fields except excluded ones
                fit_config_fields = {k: v for k, v in fit_cfg.__dict__.items() 
                                   if not k.startswith('_') and k not in EXCLUDED_FIT_PARAMS}
            elif isinstance(fit_cfg, dict):
                fit_config_fields = {k: v for k, v in fit_cfg.items() 
                                   if k not in EXCLUDED_FIT_PARAMS}
        
        # Build hashable representation
        key_data = {
            "selection_row": selection_row.tolist(),
            "row_idx": row_idx,  # Include row index for deterministic caching
            "aux_names": aux_names,
            "target_task": target_task_name,
            "config": self._make_hashable(config_fields),
            "fit_config": self._make_hashable(fit_config_fields),
        }
        
        # Create stable JSON string
        json_str = json.dumps(key_data, sort_keys=True)
        
        # Hash to get fixed-length key
        hash_obj = hashlib.sha256(json_str.encode("utf-8"))
        return hash_obj.hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a given cache key."""
        # Use subdirectories to avoid too many files in one dir
        subdir = cache_key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / f"{cache_key}.pkl"
    
    def load(
        self,
        selection_row: np.ndarray,
        aux_names: List[str],
        mtl_cfg: DotDict,
        target_task_name: str,
        fit_cfg: Optional[Any] = None,
        row_idx: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Try to load cached metrics for the given parameters.
        
        Args:
            selection_row: The design matrix row (which datasets are included)
            aux_names: List of auxiliary dataset names
            mtl_cfg: Multi-task learning configuration
            target_task_name: Name of the target task
            fit_cfg: Datamodel fit configuration
            row_idx: Row index (for deterministic caching)
        
        Returns:
            Dict of metrics if cache hit, None if cache miss
        """
        if not self.enabled:
            return None
        
        cache_key = self._make_cache_key(selection_row, aux_names, mtl_cfg, target_task_name, fit_cfg, row_idx)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with cache_path.open("rb") as f:
                cached_data = pickle.load(f)
            
            # Validate that cached data has expected structure
            if not isinstance(cached_data, dict):
                return None
            
            # Optional: add version checking here if cache format changes
            metrics = cached_data.get("metrics")
            if metrics is None or not isinstance(metrics, dict):
                return None
            
            return metrics
            
        except Exception as e:
            # Cache corruption or read error - treat as cache miss
            print(f"[RowCache] Warning: failed to load cache {cache_key[:8]}...: {e}")
            return None
    
    def save(
        self,
        selection_row: np.ndarray,
        aux_names: List[str],
        mtl_cfg: DotDict,
        target_task_name: str,
        metrics: Dict[str, float],
        fit_cfg: Optional[Any] = None,
        row_idx: Optional[int] = None,
    ) -> None:
        """
        Save metrics to cache for the given parameters.
        
        Args:
            selection_row: The design matrix row (which datasets are included)
            aux_names: List of auxiliary dataset names
            mtl_cfg: Multi-task learning configuration
            target_task_name: Name of the target task
            metrics: The metrics to cache
            fit_cfg: Datamodel fit configuration
            row_idx: Row index (for deterministic caching)
        """
        if not self.enabled:
            return
        
        cache_key = self._make_cache_key(selection_row, aux_names, mtl_cfg, target_task_name, fit_cfg, row_idx)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Store with metadata for future validation/debugging
            cache_data = {
                "metrics": metrics,
                "selection_row": selection_row.tolist(),
                "row_idx": row_idx,
                "aux_names": aux_names,
                "target_task": target_task_name,
                # Store key config fields for debugging (top-level only to keep size down)
                "config_summary": {
                    k: v for k, v in [
                        ("model_name", getattr(mtl_cfg, "model_name", None)),
                        ("lr", getattr(mtl_cfg, "lr", None)),
                        ("max_steps", getattr(mtl_cfg, "max_steps", None)),
                        ("global_token_budget", getattr(mtl_cfg, "global_token_budget", None)),
                        ("seed", getattr(mtl_cfg, "seed", None)),
                        ("k_shot", getattr(mtl_cfg, "k_shot", None)),
                    ]
                    if v is not None
                },
                "fit_config_summary": {
                    k: v for k, v in [
                        ("include_fraction", getattr(fit_cfg, "include_fraction", None) if fit_cfg else None),
                        ("seed", getattr(fit_cfg, "seed", None) if fit_cfg else None),
                        ("num_rows", getattr(fit_cfg, "num_rows", None) if fit_cfg else None),
                    ]
                    if v is not None
                } if fit_cfg else {},
            }
            
            with cache_path.open("wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception as e:
            # Don't fail the main computation if cache save fails
            print(f"[RowCache] Warning: failed to save cache {cache_key[:8]}...: {e}")
    
    def clear(self) -> int:
        """
        Clear all cached results.
        
        Returns:
            Number of cache files deleted
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0
        
        count = 0
        for cache_file in self.cache_dir.rglob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                print(f"[RowCache] Warning: failed to delete {cache_file}: {e}")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        if not self.enabled or not self.cache_dir.exists():
            return {"enabled": False, "count": 0, "size_bytes": 0}
        
        cache_files = list(self.cache_dir.rglob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "enabled": True,
            "count": len(cache_files),
            "size_bytes": total_size,
            "size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }
