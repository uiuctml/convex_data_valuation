"""
Dynamic metric inference from trainer.evaluate() output.

This is used ONLY in run_from_scores.py for final evaluation metrics.
Attribution methods in run_attribution.py always use loss only.
"""
from typing import Dict, Any
import warnings


def infer_available_metrics(trainer: Any, max_samples: int = None) -> Dict[str, Any]:
    """
    Infer available metrics by calling trainer.evaluate() on a small subset.
    
    This should ONLY be used in run_from_scores.py for comprehensive
    final evaluation. Attribution methods should use get_attribution_metric()
    which always returns "loss".
    
    Args:
        trainer: Any trainer instance with an evaluate() method
        max_samples: Maximum number of samples to evaluate for metric discovery.
                     If None (default), automatically calculated based on batch size and world size
                     to ensure at least one batch can be formed across all GPUs.
    
    Returns:
        {
            "metric_names": List[str],           # e.g., ["eval_loss", "eval_accuracy", "eval_exact_match"]
            "raw_output": Dict[str, float],      # Raw trainer.evaluate() output (from probe)
            "normalized_names": List[str],       # e.g., ["loss", "accuracy", "exact_match"]
        }
    """
    try:
        # Calculate max_samples dynamically if not provided
        if max_samples is None:
            # Get per_device_eval_batch_size (default to 8 if not set)
            per_device_batch_size = getattr(trainer.args, 'per_device_eval_batch_size', 8)
            
            # Get world_size (number of GPUs/processes)
            world_size = 1
            if hasattr(trainer.args, 'world_size'):
                world_size = trainer.args.world_size
            else:
                try:
                    import torch
                    if torch.distributed.is_initialized():
                        world_size = torch.distributed.get_world_size()
                except:
                    pass
            
            # Calculate total batch size across all devices
            total_batch_size = per_device_batch_size * world_size
            
            # Use 2x the total batch size to ensure we have enough samples
            max_samples = max(total_batch_size * 2, 16)
            print(f"[MetricInference] Auto-calculated max_samples={max_samples} "
                  f"(per_device_batch_size={per_device_batch_size} × world_size={world_size} × 2)")
        
        print(f"[MetricInference] Probing trainer.evaluate() with max_samples={max_samples} to discover metrics...")
        
        # Save original settings that might affect eval
        original_max_eval = getattr(trainer.args, 'max_eval_samples', None)
        original_eval_dataset = None
        
        # Try to create a small subset of eval_dataset for fast probing
        subset_dataset = None
        if hasattr(trainer, 'eval_dataset') and trainer.eval_dataset is not None:
            original_eval_dataset = trainer.eval_dataset
            
            # Try to subset the eval dataset (works with torch Dataset and lists)
            try:
                if hasattr(original_eval_dataset, '__len__'):
                    actual_len = len(original_eval_dataset)
                    if actual_len > max_samples:
                        # Create a subset indices
                        import random
                        indices = random.sample(range(actual_len), min(max_samples, actual_len))
                        
                        # Try torch Subset first (for HF Datasets)
                        try:
                            from torch.utils.data import Subset
                            subset_dataset = Subset(original_eval_dataset, indices)
                            print(f"[MetricInference] Using subset of {len(indices)}/{actual_len} eval samples")
                        except Exception as e:
                            warnings.warn(f"Could not create Subset: {e}")
                            subset_dataset = None
            except Exception as e:
                warnings.warn(f"Could not subset eval_dataset: {e}")
        
        # Call evaluate with the subset as a parameter (don't modify trainer.eval_dataset)
        # This way the original eval_dataset is not affected
        if subset_dataset is not None:
            print(f"[MetricInference] Evaluating with subset_dataset of length {len(subset_dataset)}")
            raw_metrics = trainer.evaluate(eval_dataset=subset_dataset)
        else:
            # Fallback: use max_eval_samples arg
            print(f"[MetricInference] No subset created, using max_eval_samples={max_samples}")
            original_max_eval = getattr(trainer.args, 'max_eval_samples', None)
            try:
                if hasattr(trainer, 'args') and hasattr(trainer.args, '__dict__'):
                    trainer.args.max_eval_samples = max_samples
                raw_metrics = trainer.evaluate()
            finally:
                # Restore original max_eval_samples
                if hasattr(trainer, 'args') and hasattr(trainer.args, '__dict__'):
                    if original_max_eval is None:
                        if hasattr(trainer.args, 'max_eval_samples'):
                            delattr(trainer.args, 'max_eval_samples')
                    else:
                        trainer.args.max_eval_samples = original_max_eval
        
        print(f"[MetricInference] Raw metrics from probe: {raw_metrics}")
        
        # Handle lm-eval nested structure: {"results": {"task_name": {"metric": value, ...}}}
        # vs standard trainer format: {"eval_loss": value, "eval_accuracy": value, ...}
        if "results" in raw_metrics and isinstance(raw_metrics["results"], dict):
            # lm-eval format: extract metrics from nested task results
            print(f"[MetricInference] Detected lm-eval format, extracting nested metrics...")
            metric_names = set()
            for task_name, task_metrics in raw_metrics["results"].items():
                if isinstance(task_metrics, dict):
                    for metric_key, metric_value in task_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            # Skip stderr metrics (format: "metric_stderr,variant") and alias
                            # The metric name part is before the comma, check if it contains "_stderr"
                            metric_base = metric_key.split(",")[0] if "," in metric_key else metric_key
                            if "_stderr" not in metric_base and metric_key != "alias":
                                metric_names.add(metric_key)
            metric_names = sorted(metric_names)
            print(f"[MetricInference] Extracted lm-eval metrics: {metric_names}")
        else:
            # Standard trainer format: metrics at top level
            metric_names = sorted([k for k, v in raw_metrics.items() 
                                  if isinstance(v, (int, float))])
        
        # Normalize metric names (strip "eval_" prefix)
        normalized = [_normalize_metric_name(m) for m in metric_names]
        
        print(f"[MetricInference] Discovered metrics: {normalized} (from keys: {metric_names})")
        
        return {
            "metric_names": metric_names,
            "raw_output": raw_metrics,
            "normalized_names": normalized,
        }
        
    except Exception as e:
        warnings.warn(f"Failed to infer metrics from trainer: {e}")
        # Fallback to just loss
        return {
            "metric_names": ["eval_loss"],
            "raw_output": {},
            "normalized_names": ["loss"],
        }


def _normalize_metric_name(metric_name: str) -> str:
    """
    Normalize metric names by stripping common prefixes.
    
    Examples:
        eval_loss -> loss
        eval_accuracy -> accuracy
        eval_exact_match -> exact_match
    """
    prefixes = ["eval_", "test_", "valid_"]
    for prefix in prefixes:
        if metric_name.startswith(prefix):
            return metric_name[len(prefix):]
    return metric_name


def get_attribution_metric() -> str:
    """
    Return the canonical metric name for attribution scoring.
    
    This is ALWAYS "loss" regardless of model/task type.
    All attribution methods in run_attribution.py must use this.
    """
    return "loss"
