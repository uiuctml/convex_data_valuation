# multisource_data/attribution/datamodel.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import Lasso

from attribution.base import DataAttribution
from attribution.save import ResultStore, AttributionRunInfo
from attribution.datamodel.cache import RowCache

# ---------------- RNG helper (prefix-invariant per-row) ----------------
def _row_rng(seed: int, row_idx: int) -> np.random.Generator:
    """
    Stateless RNG tied to (seed, row_idx). Ensures rows [0..R-1] are identical
    no matter how many rows you request overall (prefix-invariance).
    """
    ss = np.random.SeedSequence([int(seed), int(row_idx)])
    return np.random.default_rng(ss)


# ---------------- Configs ----------------
@dataclass
class DataModelFitCfg:
    """
    Controls the data-model construction & fit.
    
    Metrics are extracted and aligned by extract_metrics_from_eval().
    """
    num_rows: int = 64                   # number of sampled rows (experiments)
    alpha: float = 1e-3                  # LASSO lambda
    fit_intercept: bool = True
    normalize_columns: bool = False      # z-score columns of A before fit
    # Sampling controls (used by concrete subclasses via _sample_row)
    include_fraction: float = 0.5        # (used by uniform) fraction of auxiliaries to include per row
    seed: int = 42                       # RNG seed for row sampling
    


# ---------------- Base Class ----------------
class DataModelAttributionBase(DataAttribution, ABC):
    """
    Abstract base for data-model attribution.

    Subclasses MUST implement:
      - _sample_row(...): return a single row (np.ndarray, length 1 + N_aux).
        Column 0 must be the target indicator (usually 1.0). Aux entries follow
        the subclass's sampling rule (e.g., {0,1} for uniform, {±√(3/n),0} for CS).
      - measure_row(...): run the training/eval for a single row and return a single-
        metric dict (e.g., {"metric": 0.84}). The metric value comes from 
        extract_metrics_from_eval() which handles alignment internally.

    This base implements:
      - prefix-invariant, incremental build_design_matrix (can extend/shrink deterministically)
      - orchestration over rows (score_auxiliary_datasets)
      - LASSO fitting on aligned metric values
      - saving artifacts (A, Y_aligned, coef, scores, run_info)
    """

    # ----- Abstract hooks -----
    @abstractmethod
    def _sample_row(
        self,
        *,
        n_aux: int,
        seed: int,
        row_idx: int,
        fit_cfg: DataModelFitCfg,
    ) -> np.ndarray:
        """
        Return a single design row of length (1 + n_aux):
          row[0] = 1.0 (target fixed)
          row[1:] = aux entries per subclass rule
        Must be deterministic given (seed, row_idx).
        """

    @abstractmethod
    def measure_row(
        self,
        *,
        selection_row: np.ndarray,         # one row of A (length 1 + N_aux)
        aux_names: List[str],
    ) -> Dict[str, float]:
        """
        Run the training/eval for this row and return a single metric, e.g.:
            {"metric": <float>}
        
        The metric value comes from extract_metrics_from_eval() which returns either:
        - loss (default)
        - aligned average of multiple lm-eval metrics (if metric_specs provided)
        
        Target task and data module are accessed via self.trainer_factory.
        
        IMPORTANT: Use row_idx to set a unique seed for training to ensure
        different rows have different training trajectories!
        """

    # ----- Prefix-invariant, incremental design matrix -----
    def build_design_matrix(
        self,
        *,
        aux_names: List[str],
        fit_cfg: DataModelFitCfg,
        target_col_index: int = 0,
        existing_A: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build A with shape [R, 1 + N_aux]; column 0 is the target (usually ones).

        Prefix-invariant + incremental:
          - If `existing_A` is provided and has matching width, we reuse its prefix.
          - If requested R <= existing_R: return prefix.
          - Else append new rows deterministically using per-row RNG (seed,row_idx).
        """
        n_aux = len(aux_names)
        R = int(fit_cfg.num_rows)
        D = 1 + n_aux

        if R <= 0:
            return np.zeros((0, D), dtype=np.float32)

        # reuse prefix if possible
        if existing_A is not None:
            if existing_A.shape[1] != D:
                raise ValueError("existing_A width mismatch (aux set/order changed).")
            existing_R = existing_A.shape[0]
            if R <= existing_R:
                return existing_A[:R].copy()
            A = np.empty((R, D), dtype=np.float32)
            A[:existing_R] = existing_A
            start = existing_R
        else:
            A = np.empty((R, D), dtype=np.float32)
            start = 0

        # fill new rows deterministically
        for i in range(start, R):
            row = self._sample_row(n_aux=n_aux, seed=int(fit_cfg.seed), row_idx=i, fit_cfg=fit_cfg)
            if row.shape[0] != D:
                raise ValueError(f"_sample_row returned wrong length: {row.shape[0]} vs expected {D}")
            A[i] = row

        # force target column to 1.0 (subclasses should do it too, this is a safety net)
        A[:, target_col_index] = 1.0
        return A

    # ----- Shared coefficient fitting -----
    def _fit_and_aggregate_coef(
        self,
        *,
        A: np.ndarray,
        Y_aligned: np.ndarray,  # shape (R, 1) - single metric
        alpha: float,
        fit_intercept: bool,
        normalize_columns: bool,
    ) -> np.ndarray:
        """
        Fit LASSO on the design matrix A and response Y_aligned.
        
        Y_aligned is always (R, 1) since we use a single metric (either loss or 
        aligned average of lm-eval metrics).
        """
        A_fit = A.copy()
        if normalize_columns:
            std = A_fit.std(axis=0, ddof=0)
            std[std == 0] = 1.0
            A_fit = (A_fit - A_fit.mean(axis=0)) / std

        y = Y_aligned.ravel()
        model = Lasso(alpha=float(alpha), fit_intercept=bool(fit_intercept))
        model.fit(A_fit, y)
        return model.coef_.astype(np.float32)

    # ----- Refit with different LASSO parameters -----
    def refit_from_artifacts(
        self,
        *,
        artifacts_dir: str,
        fit_cfg: DataModelFitCfg,
        result_store: Optional[ResultStore] = None,
        method_name: str = "datamodel_refit",
        save_artifacts: bool = True,
    ) -> Dict[str, float]:
        """
        Load saved A, Y_aligned, aux_names from a previous run and refit with different LASSO parameters.
        
        Args:
            artifacts_dir: Path to the artifacts directory (e.g., outputs/attribution/run_dir/artifacts)
            fit_cfg: New fit configuration (only alpha, fit_intercept, normalize_columns, metric_weights used)
            result_store: Store for saving new results
            method_name: Name for the refit run
            save_artifacts: Whether to save new artifacts
            
        Returns:
            Dict of scores {aux_name: score}
        """
        from pathlib import Path
        import json
        
        artifacts_path = Path(artifacts_dir)
        
        # Load saved artifacts
        A = np.load(artifacts_path / "A.npy")
        Y_aligned = np.load(artifacts_path / "Y_aligned.npy")
        with open(artifacts_path / "aux_names.json", 'r') as f:
            aux_names = json.load(f)
        
        print(f"[Refit] Loaded artifacts from {artifacts_path}")
        print(f"[Refit] A shape: {A.shape}, Y_aligned shape: {Y_aligned.shape}")
        print(f"[Refit] Aux datasets: {aux_names}")
        print(f"[Refit] New LASSO parameters: alpha={fit_cfg.alpha}, fit_intercept={fit_cfg.fit_intercept}, "
              f"normalize_columns={fit_cfg.normalize_columns}")
        
        # Refit LASSO with new parameters
        coef = self._fit_and_aggregate_coef(
            A=A,
            Y_aligned=Y_aligned,
            alpha=fit_cfg.alpha,
            fit_intercept=fit_cfg.fit_intercept,
            normalize_columns=fit_cfg.normalize_columns,
        )
        
        # Extract scores (exclude target col 0)
        scores: Dict[str, float] = {name: float(coef[1 + j]) for j, name in enumerate(aux_names)}
        
        print(f"\n[{method_name.upper()}] Refit dataset attribution scores (sorted):")
        for name, val in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {name:<20s} : {val: .6f}")
        
        # Save new results if requested
        if save_artifacts and result_store is not None:
            # Try to load target task from run_info if available
            target_task = "target"
            run_info_path = artifacts_path.parent / "run_info.json"
            if run_info_path.exists():
                import json
                with open(run_info_path, 'r') as f:
                    run_info_data = json.load(f)
                    target_task = run_info_data.get("target_task", "target")
            
            run_dir = result_store.new_run_dir(method_name=method_name, target_task=target_task)
            
            # Save only the new coefficients and scores
            result_store.save_artifact(run_dir, "coef", coef)
            result_store.save_artifact(run_dir, "aux_names", aux_names)
            result_store.save_scores(run_dir, scores)
            
            # Save run info with reference to original artifacts
            from dataclasses import asdict
            new_run_info = AttributionRunInfo(
                method_name=method_name,
                target_task=target_task,
                aux_tasks=aux_names,
                model_name=None,
                device=str(self.device),
                extra={
                    "original_artifacts": str(artifacts_path),
                    "fit_cfg": asdict(fit_cfg),
                },
            )
            result_store.save_run_info(run_dir, new_run_info)
            print(f"\n[Refit] Results saved to {run_dir}")
        
        return scores

    # ----- Orchestration (shared) -----
    def score_auxiliary_datasets(
        self,
        *,
        fit_cfg: DataModelFitCfg,
        result_store: Optional[ResultStore],
        run_info: Optional[AttributionRunInfo] = None,
        method_name: str = "data_model",
        save_artifacts: bool = True,
        existing_A: Optional[np.ndarray] = None,   # <-- optional reuse
        cache_dir: Optional[str] = None,            # <-- cache directory for per-row results
        enable_cache: bool = True,                  # <-- whether to use caching
        metric_specs: Optional[List] = None,        # <-- LMEvalMetricSpec list for lm-eval format
    ) -> Dict[str, float]:
        import os
        
        # Get rank info - all ranks participate in training (DDP),
        # but only rank 0 orchestrates and collects results
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        
        if rank == 0:
            print(f"[DataModel rank {rank}] Running attribution computation")
            if world_size > 1:
                print(f"[DataModel] Each row's training will use DDP with {world_size} processes")
        
        target_task_name = self.factory.target_task
        aux_names = list(self.factory.aux_tasks)  # keep given order stable for caching outside
        mtl_cfg = self.factory.trainer.cfg
        
        store = result_store or ResultStore()
        run_dir = store.new_run_dir(method_name=method_name, target_task=target_task_name)

        target_col = 0
        
        # Set up metric extractor function for measure_row to use
        if metric_specs is not None:
            print(f"[DataModel] Using lm-eval metrics: {[(s.dataset, s.metric_name, s.higher_is_better) for s in metric_specs]}")
            self.metric_extractor_fn = lambda eval_output: self.factory.extract_metrics_from_eval(eval_output, metric_specs)
        else:
            print(f"[DataModel] Using default loss metric")
            self.metric_extractor_fn = None

        # Initialize cache
        if cache_dir is None:
            cache_dir = "outputs/datamodel_cache"
        row_cache = RowCache(cache_dir=cache_dir, enabled=enable_cache)
        
        if enable_cache:
            cache_stats = row_cache.get_stats()
            print(f"[RowCache] Cache enabled: {cache_stats['count']} entries, "
                  f"{cache_stats['size_mb']:.2f} MB in {cache_stats['cache_dir']}")

        # 1) Design matrix (can reuse/extend)
        A = self.build_design_matrix(
            aux_names=aux_names,
            fit_cfg=fit_cfg,
            target_col_index=target_col,
            existing_A=existing_A,
        )
        R = A.shape[0]

        # 2) Gather per-row metrics (with caching)
        # All ranks call measure_row (for DDP training), but only rank 0 uses results
        rows_metrics: List[float] = []
        cache_hits = 0
        cache_misses = 0
        
        # Track timing for 1-row training (only when cache misses)
        first_row_training_time = None
        
        for i in range(R):
            # Try to load from cache
            cached_metrics = row_cache.load(
                selection_row=A[i],
                aux_names=aux_names,
                mtl_cfg=mtl_cfg,
                target_task_name=target_task_name,
                fit_cfg=fit_cfg,
                row_idx=i,
            )
            
            if cached_metrics is not None:
                # Cache hit! Extract the single metric value
                metrics = cached_metrics
                cache_hits += 1
                metric_value = next(iter(metrics.values()))
                if rank == 0:
                    print(f"[DataModel] row {i+1}/{R} (cached) → {metric_value:.4f}")
            else:
                # Cache miss - compute expensive result (all ranks participate)
                cache_misses += 1
                
                # Time the first row training (ignoring cache)
                import time
                if first_row_training_time is None:
                    row_start_time = time.time()
                    metrics = self.measure_row(
                        selection_row=A[i],
                        aux_names=aux_names,
                    )
                    row_elapsed = time.time() - row_start_time
                    first_row_training_time = row_elapsed
                else:
                    metrics = self.measure_row(
                        selection_row=A[i],
                        aux_names=aux_names,
                    )
                
                # Only rank 0 extracts and saves the metric value
                if rank == 0:
                    metric_value = next(iter(metrics.values()))
                    
                    # Save to cache for future use
                    row_cache.save(
                        selection_row=A[i],
                        aux_names=aux_names,
                        mtl_cfg=mtl_cfg,
                        target_task_name=target_task_name,
                        metrics=metrics,
                        fit_cfg=fit_cfg,
                        row_idx=i,
                    )
                    print(f"[DataModel] row {i+1}/{R} (computed) → {metric_value:.4f}")
            
            # Only rank 0 collects metrics
            if rank == 0:
                rows_metrics.append(float(metric_value))

        # Only rank 0 continues with LASSO fitting
        if rank != 0:
            if world_size > 1:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
            return {}
        
        # Print cache statistics
        if enable_cache and R > 0:
            hit_rate = 100.0 * cache_hits / R if R > 0 else 0.0
            print(f"\n[RowCache] Statistics: {cache_hits} hits, {cache_misses} misses "
                  f"(hit rate: {hit_rate:.1f}%)")

        # 3) Convert to Y array (already aligned by extract_metrics_from_eval)
        Y_aligned = np.array(rows_metrics, dtype=np.float32)[:, None]  # shape (R, 1)

        # 4) Fit and aggregate coefficients
        import time
        lasso_start_time = time.time()
        coef = self._fit_and_aggregate_coef(
            A=A,
            Y_aligned=Y_aligned,
            alpha=fit_cfg.alpha,
            fit_intercept=fit_cfg.fit_intercept,
            normalize_columns=fit_cfg.normalize_columns,
        )
        lasso_fitting_time = time.time() - lasso_start_time

        # 5) Scores (exclude target col 0)
        scores: Dict[str, float] = {name: float(coef[1 + j]) for j, name in enumerate(aux_names)}

        print(f"\n[{method_name.upper()}] Dataset attribution scores (sorted):")
        for name, val in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {name:<20s} : {val: .6f}")

        # 6) Save artifacts
        if save_artifacts:
            store.save_artifact(run_dir, "A", A)
            store.save_artifact(run_dir, "Y_aligned", Y_aligned)
            store.save_artifact(run_dir, "coef", coef)
            store.save_artifact(run_dir, "aux_names", aux_names)
            store.save_scores(run_dir, scores)
            
            # Save timing information
            timing_info = {
                "lasso_fitting_seconds": lasso_fitting_time,
            }
            if first_row_training_time is not None:
                timing_info["first_row_training_seconds"] = first_row_training_time
            store.save_artifact(run_dir, "timing", timing_info)

            if run_info is None:
                run_info = AttributionRunInfo(
                    method_name=method_name,
                    target_task=target_task_name,
                    aux_tasks=aux_names,
                    model_name=self.factory.model_name,
                    device=str(self.device),
                    extra={
                        "num_rows": int(fit_cfg.num_rows),
                        "fit_cfg": asdict(fit_cfg),
                        "cache_enabled": enable_cache,
                        "cache_hits": cache_hits if enable_cache else 0,
                        "cache_misses": cache_misses if enable_cache else 0,
                    },
                )
            else:
                # Add cache stats to existing run_info
                if "extra" not in run_info.__dict__ or run_info.extra is None:
                    run_info.extra = {}
                run_info.extra["cache_enabled"] = enable_cache
                run_info.extra["cache_hits"] = cache_hits if enable_cache else 0
                run_info.extra["cache_misses"] = cache_misses if enable_cache else 0
            
            store.save_artifact(run_dir, "mtl_cfg", mtl_cfg)
            store.save_run_info(run_dir, run_info)

        # Synchronize with other ranks if distributed
        if world_size > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
        
        return scores
    


# --------------------------------------------------------------------------
# Minimal concrete implementation for refitting
# --------------------------------------------------------------------------
class _RefitOnlyDataModel(DataModelAttributionBase):
    """
    Minimal concrete implementation of DataModelAttributionBase used only for
    calling refit_from_artifacts(). This class should never be used for actual
    data model computation - it exists solely to provide a concrete instance
    for accessing the base class refit method, which is sampling-strategy agnostic.
    """
    def _sample_row(self, *, n_aux: int, seed: int, row_idx: int, fit_cfg: DataModelFitCfg) -> np.ndarray:
        raise NotImplementedError("_RefitOnlyDataModel should not be used for sampling")
    
    def measure_row(
        self,
        *,
        selection_row: np.ndarray,
        aux_names: List[str],
    ) -> Dict[str, float]:
        raise NotImplementedError("_RefitOnlyDataModel should not be used for measurement")