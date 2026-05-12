from __future__ import annotations
"""
Run multisource_data trainer using ranked aux scores.

PERFORMANCE OPTIMIZATION:
- Loads data module ONCE and shares it across ALL directories and k-values
- Uses direct in-process training (no process spawning overhead)
- This avoids reloading/reformatting data for each run (significant speedup: ~4-5x faster)

Example:
    python -m attribution.run_from_scores \\
        --base-cfg configs/mtl.yaml \\
        --dirs outputs/attribution/task_vector_SST-2_* \\
        -k 1 3 5 10
"""
import argparse
import json
import os
import sys
import tempfile
import yaml
from typing import Sequence, Dict, List, Optional
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_utils import load_yaml_config
from trainer.trainer_factory import make_trainer
from attribution.model_factory import make_full_data_model_factory
from attribution.metrics import (
    infer_available_metrics,
    MetricExtractor,
    get_metric_orientation,
    extract_metric,
    align_and_average_metrics,
    parse_metric_specs_from_config,
)


# ---------- helpers ----------

def load_scores(scores_path: str) -> Dict[str, float]:
    with open(scores_path, "r") as f:
        data = json.load(f)
    return {k: float(v) for k, v in data.items()}


def rank_aux(scores: Dict[str, float]) -> List[str]:
    """Sort aux tasks by descending score."""
    return [k for k, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def find_scores_json(exp_dir: str) -> Optional[str]:
    matches = list(Path(exp_dir).rglob("scores.json"))
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) == 0:
        print(f"[WARN] No scores.json found in {exp_dir}", file=sys.stderr)
        return None
    print(f"[INFO] Multiple scores.json detected in {exp_dir} (adaptive placeholder).", file=sys.stderr)
    return None


def run_multisource_direct(cfg_path: str, shared_data_module=None, tasks_override: Optional[List[str]] = None, eval_only: bool = False, eval_mode: Optional[str] = None):
    """
    Run training directly in-process, optionally reusing a pre-loaded data_module.
    This avoids subprocess overhead and data reloading.
    
    MULTI-GPU BEHAVIOR:
    - Training: All ranks participate (DDP)
    - HF Evaluation: All ranks participate (DDP)
    - LM-Eval: Only rank 0 launches external lm-eval process on specified GPU (lmeval_gpu config)
               Other ranks wait at barriers
    - Metrics Saving: Only rank 0 saves to disk
    
    Args:
        cfg_path: Path to config file
        shared_data_module: Pre-loaded data module to reuse
        tasks_override: List of tasks to use (target + selected aux tasks)
        eval_only: If True, skip training and only run evaluation (loads saved model)
        eval_mode: Which evaluation to run - 'hf', 'lmeval', 'both', or None (auto-detect)
    """
    print(f"[RUN-DIRECT] Loading config: {cfg_path}")
    cfg = load_yaml_config(cfg_path)
    
    # Pass the shared data module to make_trainer
    if shared_data_module is not None:
        print("[RUN-DIRECT] Reusing pre-loaded data module")
    if tasks_override is not None:
        print(f"[RUN-DIRECT] Using task override: {tasks_override}")
    
    trainer = make_trainer(cfg=cfg, data_module=shared_data_module, tasks_override=tasks_override)
    
    if eval_only:
        print(f"[RUN-DIRECT] Eval-only mode: Loading model from {cfg.output_dir}")
        
        # Check what type of checkpoint exists
        adapter_config_path = os.path.join(cfg.output_dir, "adapter_config.json")
        
        is_adapter = os.path.exists(adapter_config_path)
        
        if is_adapter:
            # Get the base model from the trainer
            base_model = trainer.model.model if hasattr(trainer.model, 'model') else trainer.model

            # Reload with explicit dtype and device mapping
            from peft import PeftModel
            reloaded_model = PeftModel.from_pretrained(
                base_model.base_model.model, 
                cfg.output_dir, 
                autocast_adapter_dtype=True,
                is_trainable=False,
                config=base_model.base_model.model.peft_config['default'],
            )
            # Re-assign back to the trainer
            if hasattr(trainer.model, 'model'):
                trainer.model.model = reloaded_model
            else:
                trainer.model = reloaded_model

        else:
            # Try loading full model
            print(f"[RUN-DIRECT] Attempting to load full model checkpoint")
            try:
                # Get the model class
                if hasattr(trainer.model, 'model'):
                    model_class = type(trainer.model.model)
                    print(f"[RUN-DIRECT] Loading into wrapped model of type {model_class.__name__}")
                    trainer.model.model = model_class.from_pretrained(cfg.output_dir)
                else:
                    model_class = type(trainer.model)
                    print(f"[RUN-DIRECT] Loading model of type {model_class.__name__}")
                    trainer.model = model_class.from_pretrained(cfg.output_dir)
                print(f"[RUN-DIRECT] Model loaded successfully")
            except Exception as e:
                print(f"[RUN-DIRECT] WARNING: Could not load model with from_pretrained: {e}")
                print(f"[RUN-DIRECT] Proceeding with evaluation using current model state")
    else:
        print(f"[RUN-DIRECT] Training with output_dir: {cfg.output_dir}")
        trainer.train()
    
    # Parse metric specs from config
    metric_specs = parse_metric_specs_from_config(cfg)
    
    # Infer available metrics BEFORE evaluation (to avoid side effects)
    # Skip probing when using lmeval mode - we'll get metrics from actual results
    # Also only probe on rank 0 to avoid redundant evaluations
    metric_info = None
    if not metric_specs and eval_mode != "lmeval":
        if trainer.is_world_process_zero():
            metric_info = infer_available_metrics(trainer)
        # Broadcast metric info to other ranks if distributed
        if dist.is_available() and dist.is_initialized():
            metric_info_list = [metric_info]
            dist.broadcast_object_list(metric_info_list, src=0)
            metric_info = metric_info_list[0]
    
    # Check if trainer supports multiple evaluation modes
    has_hf_eval = hasattr(trainer, 'evaluate_hf') and callable(getattr(trainer, 'evaluate_hf'))
    has_lmeval = hasattr(trainer, 'evaluate_lmeval') and callable(getattr(trainer, 'evaluate_lmeval'))
    
    all_eval_results = {}
    final_metrics = {}
    
    # Determine which evaluations to run based on eval_mode parameter
    if eval_mode is None:
        # Auto-detect: run both if available, otherwise run what's available
        run_hf = has_hf_eval
        run_lmeval = has_lmeval
    elif eval_mode == "both":
        run_hf = has_hf_eval
        run_lmeval = has_lmeval
    elif eval_mode == "hf":
        run_hf = has_hf_eval
        run_lmeval = False
    elif eval_mode == "lmeval":
        run_hf = False
        run_lmeval = has_lmeval
    else:
        raise ValueError(f"Invalid eval_mode: {eval_mode}. Expected 'hf', 'lmeval', 'both', or None")
    
    # Run evaluations based on determined flags
    if run_hf and run_lmeval:
        print("[EVAL] Running both HF and lm-eval evaluations...")
        
        print("[EVAL] Running HF evaluation...")
        hf_metrics = trainer.evaluate_hf()
        all_eval_results["hf"] = hf_metrics
        
        print("[EVAL] Running lm-eval evaluation...")
        lmeval_metrics = trainer.evaluate_lmeval()
        all_eval_results["lmeval"] = lmeval_metrics
        
        # Use lm-eval results as the primary metrics for compatibility
        final_metrics = lmeval_metrics
    elif run_hf:
        print("[EVAL] Running HF evaluation only...")
        hf_metrics = trainer.evaluate_hf()
        all_eval_results["hf"] = hf_metrics
        final_metrics = hf_metrics
    elif run_lmeval:
        print("[EVAL] Running lm-eval evaluation only...")
        lmeval_metrics = trainer.evaluate_lmeval()
        all_eval_results["lmeval"] = lmeval_metrics
        final_metrics = lmeval_metrics
    else:
        # Fallback to default evaluate() method
        print("[EVAL] Running default evaluation...")
        final_metrics = trainer.evaluate()
        all_eval_results["default"] = final_metrics
    
    if trainer.is_world_process_zero():
        print("[FINAL] Raw metrics:", final_metrics)
        
        if metric_specs:
            # Using lm-eval metrics from config
            print(f"[FINAL] Using metric specifications from config: {[(s.dataset, s.metric_name, s.higher_is_better) for s in metric_specs]}")
            
            # Extract individual metrics
            individual_metrics = {}
            for spec in metric_specs:
                key = f"{spec.dataset}:{spec.metric_name}"
                try:
                    value = extract_metric(final_metrics, metric_spec=spec)
                    individual_metrics[key] = value
                    direction = "↑" if spec.higher_is_better else "↓"
                    print(f"  - {key}: {value:.4f} {direction}")
                except Exception as e:
                    print(f"  - {key}: ERROR - {e}")
                    individual_metrics[key] = None
            
            # Compute aligned average
            try:
                aligned_avg = align_and_average_metrics(final_metrics, metric_specs)
                print(f"[FINAL] Aligned average (all normalized to ↑): {aligned_avg:.4f}")
            except Exception as e:
                print(f"[FINAL] Could not compute aligned average: {e}")
                aligned_avg = None
            
            normalized_metrics = {
                "individual_metrics": individual_metrics,
                "aligned_average": aligned_avg,
            }
        else:
                # No metric_specs - either use HF metric_info or extract from lm-eval results
                if metric_info is not None:
                    # Using default HF trainer metrics (already inferred before eval)
                    metric_extractor = MetricExtractor(metric_info["metric_names"])
                    
                    # Extract and normalize all available metrics
                    normalized_metrics = metric_extractor.extract(final_metrics)
                    print("[FINAL] Normalized metrics:", normalized_metrics)
                    
                    # Show metric orientations for reference
                    print("[FINAL] Metric orientations:")
                    for metric_name in metric_info["normalized_names"]:
                        orientation = get_metric_orientation(metric_name)
                        direction = "higher is better" if orientation > 0 else "lower is better"
                        print(f"  - {metric_name}: {direction} (orientation={orientation})")
                elif "results" in final_metrics and isinstance(final_metrics["results"], dict):
                    # lm-eval mode without metric_specs - extract metrics from results
                    print("[FINAL] Extracting metrics from lm-eval results...")
                    individual_metrics = {}
                    for task_name, task_metrics in final_metrics["results"].items():
                        if isinstance(task_metrics, dict):
                            for metric_key, metric_value in task_metrics.items():
                                if isinstance(metric_value, (int, float)):
                                    # Skip stderr metrics and alias
                                    metric_base = metric_key.split(",")[0] if "," in metric_key else metric_key
                                    if "_stderr" not in metric_base and metric_key != "alias":
                                        full_key = f"{task_name}:{metric_key}"
                                        individual_metrics[full_key] = metric_value
                                        print(f"  - {full_key}: {metric_value:.4f}")
                    
                    normalized_metrics = {
                        "individual_metrics": individual_metrics,
                        "aligned_average": None,  # Can't compute without metric_specs
                    }
                else:
                    # Fallback - just save raw metrics
                    normalized_metrics = {"raw": final_metrics}
                    print("[FINAL] No metric extraction available, saving raw metrics")
    if not eval_only:
        trainer.save_model()
    
    if trainer.is_world_process_zero():
        # Save both raw and normalized metrics
        metrics_path = os.path.join(cfg.output_dir, "final_metrics.json")
        
        # Load existing metrics if available (to preserve untouched eval results)
        existing_metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    existing_metrics = json.load(f)
                print(f"[save] Loaded existing metrics from {metrics_path}")
            except Exception as e:
                print(f"[save] Could not load existing metrics: {e}")
        
        # Merge all_eval_results with existing results (only update what was run)
        merged_eval_results = existing_metrics.get("all_eval_results", {})
        merged_eval_results.update(all_eval_results)  # Only overwrites keys that were just run
        
        metrics_to_save = {
            "raw_metrics": final_metrics,
            "normalized_metrics": normalized_metrics,
            "all_eval_results": merged_eval_results,  # Merged results preserve old untouched modes
        }
        
        # Add metric info based on whether we used metric_specs or not
        if metric_specs:
            metrics_to_save["metric_info"] = {
                "type": "lm_eval",
                "metric_specs": [(s.dataset, s.metric_name, s.higher_is_better) for s in metric_specs],
            }
        elif metric_info is not None:
            metrics_to_save["metric_info"] = {
                "type": "hf_trainer",
                "available_metrics": metric_info["normalized_names"],
                "orientations": {m: get_metric_orientation(m) for m in metric_info["normalized_names"]}
            }
        else:
            # lmeval mode without explicit metric_specs - extract from results
            metrics_to_save["metric_info"] = {
                "type": "lm_eval_auto",
                "note": "Metrics extracted from lm-eval results",
            }
        
        # Track all eval modes (both old and new)
        metrics_to_save["eval_modes_run"] = list(merged_eval_results.keys())
        
        with open(metrics_path, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"[save] Final metrics saved to {metrics_path}")
        print(f"[save] Evaluation modes in file: {list(merged_eval_results.keys())}")
        print(f"[save] Evaluation modes updated this run: {list(all_eval_results.keys())}")


def generate_temp_cfg(base_cfg_path: str, output_dir: str, target_task: str, aux_tasks: List[str]) -> str:
    """Write a temporary YAML that will be deleted automatically."""
    with open(base_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["target_task"] = target_task
    cfg["aux_tasks"] = aux_tasks
    cfg["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp_path = tmp.name
    tmp.close()
    return tmp_path


# ---------- main orchestration ----------

def main(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description="Run multisource_data trainer using ranked aux scores (non-adaptive).")
    ap.add_argument("--base-cfg", type=str, required=True, help="Path to base YAML config template.")
    ap.add_argument("--dirs", nargs="+", required=True, help="Experiment directories containing scores.json.")
    ap.add_argument("-k", "--k-values", nargs="+", type=int, required=True, help="List of k values.")
    ap.add_argument("--target", type=str, default=None, help="Target task name override (else inferred).")
    ap.add_argument("--dry-run", action="store_true", help="Print plan without launching training.")
    ap.add_argument("--force", action="store_true", help="Force retraining even if final_metrics.json exists.")
    ap.add_argument("--eval-only", action="store_true", help="Skip training, load saved models and re-run evaluation only.")
    ap.add_argument("--eval-mode", type=str, default=None, choices=["hf", "lmeval", "both"], 
                    help="Which evaluation to run: 'hf' (HuggingFace), 'lmeval' (lm-evaluation-harness), 'both', or None (auto-detect all available).")
    args = ap.parse_args(argv)

    # Determine if we're in a distributed setting
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    is_main_process = (rank == 0)
    
    if is_distributed:
        print(f"[RunFromScores] Running in distributed mode: rank {rank}/{world_size}")
    else:
        print(f"[RunFromScores] Running in single-process mode")

    # Pre-load data module ONCE for all directories and k-values
    shared_data_module = None
    if not args.dry_run:
        if is_main_process:
            print("[INFO] Pre-loading data module (will be shared across ALL directories and k-values)...")
        
        # Collect all unique tasks across all directories
        all_tasks = set()
        
        # Add target task explicitly if provided
        if args.target:
            all_tasks.add(args.target)
            if is_main_process:
                print(f"[INFO] Target task specified: {args.target}")
        
        for exp_dir in args.dirs:
            exp_dir = os.path.abspath(exp_dir)
            scores_path = find_scores_json(exp_dir)
            if not scores_path:
                if is_main_process:
                    print(f"[INFO] Skipping {exp_dir} (no scores.json)")
                continue
            
            scores = load_scores(scores_path)
            ranked = rank_aux(scores)
            
            # Infer target from directory name if not provided
            inferred_target = (
                args.target
                or (Path(exp_dir).name.split("_")[0] if "_" in Path(exp_dir).name else Path(exp_dir).name)
            )
            all_tasks.add(inferred_target)
            
            # Add all auxiliary tasks from scores
            all_tasks.update(ranked)
            if is_main_process:
                print(f"[INFO] Found {len(ranked)} aux tasks in {Path(exp_dir).name}")
        
        if not all_tasks:
            raise ValueError("No tasks found! Check that --dirs contain valid scores.json files.")
        
        all_tasks_list = sorted(all_tasks)
        if is_main_process:
            print(f"[INFO] Total unique tasks to load: {len(all_tasks_list)}")
            print(f"[INFO] Tasks: {all_tasks_list}")
        
        # Load shared data module for FULL datasets (no k-shot) via the factory
        # Use the same config loading path as the rest of the code
        base_cfg = load_yaml_config(args.base_cfg)
        # Build a factory configured to load FULL datasets for these tasks
        factory = make_full_data_model_factory(base_cfg)
        # Reuse the underlying data module across all training runs
        shared_data_module = factory.data_module
        
        # Synchronize after data loading (ensure all ranks have loaded)
        if is_distributed:
            dist.barrier()
            if is_main_process:
                print("[INFO] All ranks synchronized after data module loading")

    for exp_dir in args.dirs:
        exp_dir = os.path.abspath(exp_dir)
        scores_path = find_scores_json(exp_dir)
        if not scores_path:
            continue

        scores = load_scores(scores_path)
        ranked = rank_aux(scores)
        target = args.target or Path(exp_dir).name

        if is_main_process:
            print(f"\n[INFO] Processing {exp_dir}")
            print(f"[INFO] Target: {target}")
            print(f"[INFO] {len(ranked)} aux tasks: {ranked}")

        # Check if all scores are non-positive (special case: train target-only)
        positive_tasks = [task for task in ranked if scores[task] > 0]
        all_scores_nonpositive = len(positive_tasks) == 0
        
        if all_scores_nonpositive:
            if is_main_process:
                print(f"[WARN] All attribution scores are non-positive!")
                print(f"[WARN] Will train target-only model for smallest k and copy to all k values")
            
            # Train target-only model for smallest k
            smallest_k = min(args.k_values)
            target_only_dir = os.path.join(exp_dir, f"train_k{smallest_k}")
            
            # Check if already trained
            metrics_path = os.path.join(target_only_dir, "final_metrics.json")
            if os.path.exists(metrics_path) and not args.force and not args.eval_only:
                if is_main_process:
                    print(f"[INFO] Target-only model already exists at {target_only_dir}")
            else:
                if is_main_process:
                    print(f"[PLAN] Training target-only model: k={smallest_k}, output={target_only_dir}")
                
                if not args.dry_run:
                    # Only rank 0 generates config to avoid race conditions
                    tmp_cfg = None
                    if is_main_process:
                        # Load original config to compute adjusted max_steps
                        with open(args.base_cfg, "r") as f:
                            base_cfg_dict = yaml.safe_load(f)
                        
                        original_max_steps = base_cfg_dict.get("max_steps", 1000)
                        target_weight = base_cfg_dict.get("target_weight", 1.0)
                        adjusted_max_steps = int(original_max_steps * target_weight)
                        
                        print(f"[INFO] Adjusted max_steps for target-only: {original_max_steps} * {target_weight} = {adjusted_max_steps}")
                        
                        # Generate temporary YAML with target-only settings
                        tmp_cfg = generate_temp_cfg(args.base_cfg, target_only_dir, target, [])
                        
                        # Load and modify the temp config
                        with open(tmp_cfg, "r") as f:
                            tmp_cfg_dict = yaml.safe_load(f)
                        
                        # Override for target-only training
                        tmp_cfg_dict["max_steps"] = adjusted_max_steps
                        tmp_cfg_dict["target_weight"] = 1.0
                        tmp_cfg_dict["aux_weight_total"] = 0.0
                        
                        # Save modified config
                        with open(tmp_cfg, "w") as f:
                            yaml.safe_dump(tmp_cfg_dict, f)
                    
                    # Broadcast config path to all ranks
                    if is_distributed:
                        tmp_cfg_list = [tmp_cfg] if is_main_process else [None]
                        dist.broadcast_object_list(tmp_cfg_list, src=0)
                        tmp_cfg = tmp_cfg_list[0]
                    
                    try:
                        # Train with only target task, using mix_override to ensure target_weight=1.0
                        # All ranks participate in training
                        run_multisource_direct(
                            tmp_cfg,
                            shared_data_module=shared_data_module,
                            tasks_override=[target],
                            eval_only=args.eval_only,
                            eval_mode=args.eval_mode
                        )
                    finally:
                        # Only rank 0 cleans up temporary YAML
                        if is_main_process and tmp_cfg and os.path.exists(tmp_cfg):
                            os.remove(tmp_cfg)
                            if os.path.exists(tmp_cfg + "~"):
                                os.remove(tmp_cfg + "~")
            
            # Copy model to all other k values
            if not args.dry_run and is_main_process:
                import shutil
                for k in args.k_values:
                    if k == smallest_k:
                        continue
                    
                    dest_dir = os.path.join(exp_dir, f"train_k{k}")
                    dest_metrics_path = os.path.join(dest_dir, "final_metrics.json")
                    
                    # Skip if already exists and not forcing
                    if os.path.exists(dest_metrics_path) and not args.force:
                        print(f"[SKIP] k={k}, output={dest_dir} (already exists, use --force to overwrite)")
                        continue
                    
                    print(f"[COPY] Copying target-only model from k={smallest_k} to k={k}")
                    
                    # Remove destination if it exists
                    if os.path.exists(dest_dir):
                        shutil.rmtree(dest_dir)
                    
                    # Copy entire directory
                    shutil.copytree(target_only_dir, dest_dir)
                    print(f"[COPY] Completed: {dest_dir}")
            
            # Synchronize after copying (if distributed)
            if is_distributed:
                dist.barrier()
            
            # Skip normal k-value processing for this directory
            continue

        for k in args.k_values:
            if k < 1:
                continue
            k = min(k, len(ranked))
            topk = ranked[:k]
            # Filter out tasks with non-positive scores
            topk = [task for task in topk if scores[task] > 0]
            
            # Skip if no positive-scored tasks remain (shouldn't happen due to check above)
            if not topk:
                if is_main_process:
                    print(f"[SKIP] k={k}, no tasks with positive scores")
                continue
            
            out_dir = os.path.join(exp_dir, f"train_k{k}")
            
            # Check if training already completed (skip unless --force or --eval-only)
            metrics_path = os.path.join(out_dir, "final_metrics.json")
            if os.path.exists(metrics_path) and not args.force and not args.eval_only:
                if is_main_process:
                    print(f"[SKIP] k={k}, output={out_dir} (final_metrics.json already exists, use --force to retrain)")
                continue
            
            # In eval-only mode, check if model exists
            if args.eval_only:
                model_exists = (
                    os.path.exists(os.path.join(out_dir, "pytorch_model.bin")) or
                    os.path.exists(os.path.join(out_dir, "adapter_model.bin")) or
                    os.path.exists(os.path.join(out_dir, "model.safetensors")) or
                    os.path.exists(os.path.join(out_dir, "adapter_model.safetensors"))
                )
                if not model_exists:
                    if is_main_process:
                        print(f"[SKIP] k={k}, output={out_dir} (--eval-only mode but no model checkpoint found)")
                    continue
            
            actual_k = len(topk)
            if is_main_process:
                print(f"[PLAN] k={k} (actual={actual_k} after filtering score>0), aux={topk}, output={out_dir}")

            if args.dry_run:
                continue

            # Only rank 0 generates temporary YAML to avoid race conditions
            tmp_cfg = None
            if is_main_process:
                tmp_cfg = generate_temp_cfg(args.base_cfg, out_dir, target, topk)
            
            # Broadcast config path to all ranks
            if is_distributed:
                tmp_cfg_list = [tmp_cfg] if is_main_process else [None]
                dist.broadcast_object_list(tmp_cfg_list, src=0)
                tmp_cfg = tmp_cfg_list[0]
            
            try:
                # Pass tasks_override: target + top-k aux tasks
                # All ranks participate in training/evaluation
                tasks_for_this_run = [target] + topk
                run_multisource_direct(
                    tmp_cfg, 
                    shared_data_module=shared_data_module, 
                    tasks_override=tasks_for_this_run,
                    eval_only=args.eval_only,
                    eval_mode=args.eval_mode
                )
            finally:
                # Only rank 0 cleans up temporary YAML
                if is_main_process and tmp_cfg and os.path.exists(tmp_cfg):
                    os.remove(tmp_cfg)
                    if os.path.exists(tmp_cfg + "~"):
                        os.remove(tmp_cfg + "~")  # some editors leave backups

        # Placeholder for adaptive (multiple rounds)
        # raise NotImplementedError("Adaptive flow not implemented yet.")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Properly cleanup distributed process group if initialized
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            print("[RunFromScores] Distributed process group destroyed")