from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch.distributed as dist

"""
Minimal CLI for dataset attribution runs.

Args:
  --config : path to YAML experiment config (shared with MTL + TaskVector)
  --k_shot : number of examples for each auxiliary dataset
  --method : one_step | task_vector | kmm | uniform_data_model | cs_data_model |
             datamodel_refit

Optional (task_vector & one_step):
  --similarity : cos | dot (default: cos)

Optional (kmm - assumes precomputed gradients):
  --kmm_artifacts_dir : path to directory with precomputed gradients (REQUIRED for kmm)
  --kmm_lambda        : L1 regularization coefficient (default: 1e-2)
  --kmm_solver        : CVXPY solver (OSQP | ECOS | SCS, default: OSQP)
  --kmm_source        : gradient source type (task_vector | one_step, default: task_vector)
  --kmm_max_iter      : maximum solver iterations (default: 20000)

Optional (data_model for both uniform & CS):
  --num_rows          : number of sampled rows (default: 64)
  --alpha             : LASSO regularization coefficient (default: 1e-3)
  --include_fraction  : fraction of auxiliaries per row (default: 0.5, uniform only)
  --dm_seed           : RNG seed (default: cfg.seed)
  --fit_intercept     : fit intercept in regression (flag)
  --normalize_columns : normalize A's columns before fitting (flag)

Optional (datamodel_refit):
  --artifacts_dir : path to artifacts directory for refitting (REQUIRED for datamodel_refit)

Optional (debug):
  --debug : enable debug mode to validate projection/lifting correctness (flag)

Config-based (metric configuration for all methods):
  metric_specs : List of metric specifications in config YAML
                 Format:
                   metric_specs:
                     - dataset: gsm8k
                       metric_name: exact_match,flexible-extract
                       higher_is_better: true
                 If not provided, defaults to loss (for HF trainer)
"""

import argparse
import os
from pathlib import Path
import yaml

from attribution.gradient.one_step import OneStepGradientAttribution
from attribution.gradient.task_vector import TaskVectorAttribution
from attribution.gradient.kmm import GradientKMMAttribution

from attribution.datamodel.base import DataModelFitCfg, _RefitOnlyDataModel
from attribution.datamodel.uniform import UniformSamplingDataModel
from attribution.datamodel.compressed_sensing import CompressedSensingDataModel

from attribution.save import ResultStore, AttributionRunInfo
from attribution.metrics import parse_metric_specs_from_config

from utils.config_utils import DotDict
from attribution.model_factory import make_aux_kshot_model_factory


# --------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML MTL config file.")
    p.add_argument("--k_shot", type=int, required=True,
                   help="Number of examples for each auxiliary dataset.")
    p.add_argument("--method", type=str, required=True,
                   choices=["one_step", "task_vector", "kmm",
                            "uniform_data_model", "cs_data_model",
                            "datamodel_refit"],
                   help="Attribution method to run.")
    p.add_argument("--similarity", type=str, default="cos",
                   choices=["cos", "dot"],
                   help="Similarity metric for scoring (task_vector & one_step).")

    # KMM specific (assumes precomputed gradients)
    p.add_argument("--kmm_artifacts_dir", type=str, default=None,
                   help="Path to directory with precomputed gradients (task_vector or one_step artifacts).")
    p.add_argument("--kmm_lambda", type=float, default=1e-2,
                   help="L1 regularization coefficient for KMM.")
    p.add_argument("--kmm_solver", type=str, default="OSQP",
                   choices=["OSQP", "ECOS", "SCS"],
                   help="CVXPY solver for KMM optimization.")
    p.add_argument("--kmm_source", type=str, default="task_vector",
                   choices=["task_vector", "one_step"],
                   help="Source of precomputed gradients (default: task_vector).")
    p.add_argument("--kmm_max_iter", type=int, default=20000,
                   help="Maximum iterations for KMM solver.")

    # Data-model specific (used by both uniform & CS; include_fraction is ignored by CS)
    p.add_argument("--num_rows", type=int, default=64,
                   help="Number of sampled rows.")
    p.add_argument("--alpha", type=float, default=1e-3,
                   help="LASSO regularization coefficient.")
    p.add_argument("--include_fraction", type=float, default=0.5,
                   help="(Uniform only) Fraction of auxiliaries to include per row.")
    p.add_argument("--dm_seed", type=int, default=None,
                   help="RNG seed for sampling (defaults to cfg.seed).")
    p.add_argument("--fit_intercept", action="store_true",
                   help="Include intercept in regression.")
    p.add_argument("--normalize_columns", action="store_true",
                   help="Normalize design-matrix columns before fitting.")
    # Refit-specific
    p.add_argument("--artifacts_dir", type=str, default=None,
                   help="Path to artifacts directory for refitting (required for method=datamodel_refit).")

    # Debug mode
    p.add_argument("--debug", action="store_true",
                help="Enable debug mode to validate projection/lifting correctness.")
    
    return p.parse_args()


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return DotDict(yaml.safe_load(f))


# --------------------------------------------------------------------------
def main():
    args = _parse_args()
    cfg = _load_yaml(args.config)
    
    # Inject CLI k_shot into config so it's available to trainer
    cfg["k_shot"] = args.k_shot

    # Core config
    model_name  = cfg["model_name"]
    target_task = cfg["target_task"]
    all_tasks   = cfg["tasks"]
    aux_tasks   = [t for t in all_tasks if t != target_task]
    output_dir  = cfg["output_dir"]

    # Parse metric specifications from config (if provided)
    metric_specs = parse_metric_specs_from_config(cfg)
    if metric_specs:
        print(f"[Attribution] Using metric specifications from config: {[(s.dataset, s.metric_name, s.higher_is_better) for s in metric_specs]}")
    else:
        print(f"[Attribution] Using default metric (loss)")

    # Result directory
    results_root = Path(output_dir) / "attribution"
    results_root.mkdir(parents=True, exist_ok=True)
    store = ResultStore(root_dir=results_root)
    factory = make_aux_kshot_model_factory(cfg, aux_k_shot=args.k_shot, debug_mode=args.debug)
    device = str(factory.trainer.args.device)

    # ----------------------------------------------------------------------
    # ONE-STEP
    # ----------------------------------------------------------------------
    if args.method == "one_step":
        attrib = OneStepGradientAttribution(
            trainer_factory=factory,
        )

        # Build run_info specific to one_step with similarity metric
        run_info = AttributionRunInfo(
            method_name="one_step",
            target_task=target_task,
            aux_tasks=aux_tasks,
            model_name=model_name,
            device=device,
            extra={
                "k_shot": args.k_shot,
                "similarity": args.similarity,
                "config": os.path.basename(args.config),
            },
        )

        attrib.score_auxiliary_datasets(
            similarity=args.similarity,
            result_store=store,
            run_info=run_info,
            save_artifacts=True,
        )

    # ----------------------------------------------------------------------
    # TASK VECTOR
    # ----------------------------------------------------------------------
    elif args.method == "task_vector":
        attrib = TaskVectorAttribution(
            trainer_factory=factory,
        )

        # Build run_info specific to task_vector with similarity metric
        run_info = AttributionRunInfo(
            method_name="task_vector",
            target_task=target_task,
            aux_tasks=aux_tasks,
            model_name=model_name,
            device=device,
            extra={
                "k_shot": args.k_shot,
                "similarity": args.similarity,
                "config": os.path.basename(args.config),
            },
        )

        attrib.score_auxiliary_datasets(
            result_store=store,
            run_info=run_info,
            similarity=args.similarity,
            save_artifacts=True,
        )

    # ----------------------------------------------------------------------
    # KMM (assumes precomputed gradients from task_vector or one_step)
    # ----------------------------------------------------------------------
    elif args.method == "kmm":
        if args.kmm_artifacts_dir is None:
            raise ValueError(
                "--kmm_artifacts_dir is required for method=kmm. "
                "Provide path to task_vector or one_step artifacts directory."
            )
        attrib = GradientKMMAttribution(
            trainer_factory=factory,
            artifact_dir=args.kmm_artifacts_dir,
            source_preference=[args.kmm_source],  # "task_vector" or "one_step"
            l1_lambda=args.kmm_lambda,
            solver=args.kmm_solver,
            solver_kwargs={"max_iter": args.kmm_max_iter},
            verbose=True,
        )

        run_info = AttributionRunInfo(
            method_name="kmm",
            target_task=target_task,
            aux_tasks=aux_tasks,
            model_name=model_name,
            device=device,
            extra={
                "k_shot": args.k_shot,
                "config": os.path.basename(args.config),
                "kmm_lambda": args.kmm_lambda,
                "kmm_solver": args.kmm_solver,
                "kmm_source": args.kmm_source,
                "artifacts_dir": args.kmm_artifacts_dir,
            },
        )

        _ = attrib.score_auxiliary_datasets(
            result_store=store,
            run_info=run_info,
            save_artifacts=True,
        )

    # ----------------------------------------------------------------------
    # DATA MODEL (UniformSamplingDataModel)
    # ----------------------------------------------------------------------
    elif args.method == "uniform_data_model":
        # Build fit configuration
        seed = args.dm_seed or cfg.get("seed", 42)
        fit_cfg = DataModelFitCfg(
            num_rows=int(args.num_rows),
            alpha=float(args.alpha),
            include_fraction=float(args.include_fraction),
            seed=int(seed),
            fit_intercept=bool(args.fit_intercept),
            normalize_columns=bool(args.normalize_columns),
        )
        print(f"[DataModel-Uniform] Using configuration: {fit_cfg}")

        attrib = UniformSamplingDataModel(trainer_factory=factory)

        _ = attrib.score_auxiliary_datasets(
            fit_cfg=fit_cfg,
            result_store=store,
            run_info=AttributionRunInfo(
                method_name="data_model_uniform",
                target_task=target_task,
                aux_tasks=aux_tasks,
                model_name=model_name,
                device=device,
                extra={
                    "k_shot": args.k_shot,
                    "config": os.path.basename(args.config),
                    "metric": "loss" if metric_specs is None else "lm_eval_multi",
                    "fit_cfg": vars(fit_cfg),
                    "metric_specs": [(s.dataset, s.metric_name, s.higher_is_better) for s in metric_specs] if metric_specs else None,
                },
            ),
            method_name="data_model_uniform",
            save_artifacts=True,
            metric_specs=metric_specs,
        )

    # ----------------------------------------------------------------------
    # DATA MODEL (CompressedSensingDataModel)
    # ----------------------------------------------------------------------
    elif args.method == "cs_data_model":
        # Build fit configuration
        # Note: include_fraction is ignored by CS
        seed = args.dm_seed or cfg.get("seed", 42)
        fit_cfg = DataModelFitCfg(
            num_rows=int(args.num_rows),
            alpha=float(args.alpha),
            include_fraction=float(args.include_fraction),  # ignored by CS
            seed=int(seed),
            fit_intercept=bool(args.fit_intercept),
            normalize_columns=bool(args.normalize_columns),
        )
        print(f"[DataModel-CS] Using configuration: {fit_cfg}")

        attrib = CompressedSensingDataModel(trainer_factory=factory)

        _ = attrib.score_auxiliary_datasets(
            fit_cfg=fit_cfg,
            result_store=store,
            run_info=AttributionRunInfo(
                method_name="data_model_cs",
                target_task=target_task,
                aux_tasks=aux_tasks,
                model_name=model_name,
                device=device,
                extra={
                    "k_shot": args.k_shot,
                    "config": os.path.basename(args.config),
                    "metric": "loss" if metric_specs is None else "lm_eval_multi",
                    "fit_cfg": vars(fit_cfg),
                    "metric_specs": [(s.dataset, s.metric_name, s.higher_is_better) for s in metric_specs] if metric_specs else None,
                },
            ),
            method_name="data_model_cs",
            save_artifacts=True,
            metric_specs=metric_specs,
        )
    # ----------------------------------------------------------------------
    # REFIT (reuse saved A and Y_aligned with different LASSO parameters)
    # ----------------------------------------------------------------------
    elif args.method == "datamodel_refit":
        if args.artifacts_dir is None:
            raise ValueError("--artifacts_dir is required for method=datamodel_refit")

        # Build new fit configuration
        seed = args.dm_seed or cfg.get("seed", 42)
        fit_cfg = DataModelFitCfg(
            num_rows=int(args.num_rows),             # not used by refit, but kept for completeness
            alpha=float(args.alpha),
            include_fraction=float(args.include_fraction),  # not used by refit
            seed=int(seed),                          # not used by refit
            fit_intercept=bool(args.fit_intercept),
            normalize_columns=bool(args.normalize_columns),
        )
        print(f"[DataModel-Refit] Using configuration: {fit_cfg}")

        attrib = _RefitOnlyDataModel(trainer_factory=factory)

        _ = attrib.refit_from_artifacts(
            artifacts_dir=args.artifacts_dir,
            fit_cfg=fit_cfg,
            result_store=store,
            method_name="datamodel_refit",
            save_artifacts=True,
        )
    else:
        raise ValueError(f"Unsupported attribution method: {args.method}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Properly cleanup distributed process group if initialized
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            print("[Attribution] Distributed process group destroyed")