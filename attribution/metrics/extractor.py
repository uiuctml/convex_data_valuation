"""
Metric extraction and normalization utilities.

Used to extract metrics from trainer.evaluate() output, handling
various naming conventions (eval_loss vs loss, etc.) and lm-eval format.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import warnings
import numpy as np


@dataclass
class LMEvalMetricSpec:
    """
    Specification for extracting a metric from lm-eval results.
    
    Args:
        dataset: Dataset name (e.g., "gsm8k", "hendrycks_math")
        metric_name: Metric name within the dataset (e.g., "exact_match,flexible-extract")
        higher_is_better: True if higher values are better, False if lower is better
    """
    dataset: str
    metric_name: str
    higher_is_better: bool = True


class MetricExtractor:
    """
    Handles extraction and normalization of metrics from trainer.evaluate() output.
    
    This is used ONLY in run_from_scores.py for comprehensive final evaluation.
    Attribution methods should use extract_metric() or extract_loss() directly.
    """
    
    def __init__(self, available_metrics: List[str]):
        """
        Args:
            available_metrics: List of raw metric names (e.g., ["eval_loss", "eval_accuracy"])
        """
        self.available_metrics = available_metrics
        self._build_lookup_map()
    
    def _build_lookup_map(self):
        """Build mapping from normalized names to raw names."""
        self.lookup_map: Dict[str, str] = {}
        for raw_name in self.available_metrics:
            normalized = self._normalize(raw_name)
            self.lookup_map[normalized] = raw_name
    
    def _normalize(self, metric_name: str) -> str:
        """Strip eval_ prefix."""
        prefixes = ["eval_", "test_", "valid_"]
        for prefix in prefixes:
            if metric_name.startswith(prefix):
                return metric_name[len(prefix):]
        return metric_name
    
    def extract(self, eval_output: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract and normalize all available metrics from evaluation output.
        
        This should ONLY be used in run_from_scores.py for final evaluation.
        
        Args:
            eval_output: Raw trainer.evaluate() output
        
        Returns:
            Dict with normalized metric names -> values
        """
        result = {}
        for normalized, raw_name in self.lookup_map.items():
            if raw_name in eval_output:
                value = eval_output[raw_name]
                if isinstance(value, (int, float)):
                    result[normalized] = float(value)
                else:
                    warnings.warn(f"Metric {raw_name} has non-numeric value: {value}")
            else:
                # Metric not in output (shouldn't happen if we inferred correctly)
                result[normalized] = None
        
        return result


def extract_metric(
    eval_output: Dict[str, Any],
    metric_spec: Optional[LMEvalMetricSpec] = None,
    metric_name: str = "loss",
) -> float:
    """
    Extract a metric from evaluation output.
    
    Supports two formats:
    1. HF Trainer format: {"eval_loss": 1.5, "eval_accuracy": 0.8, ...}
    2. lm-eval format: {"results": {"gsm8k": {"exact_match,flexible-extract": 0.5, ...}}}
    
    Args:
        eval_output: Raw evaluation output (from trainer.evaluate() or lm-eval)
        metric_spec: LMEvalMetricSpec for lm-eval format (if None, use HF format)
        metric_name: Metric name for HF format (only used if metric_spec is None)
    
    Returns:
        Metric value
    
    Raises:
        ValueError: If metric cannot be found in output
    """
    # Case 1: lm-eval format
    if metric_spec is not None:
        if "results" not in eval_output:
            raise ValueError(
                f"lm-eval format expected (with 'results' key), but got keys: {list(eval_output.keys())}"
            )
        
        results = eval_output["results"]
        
        # Handle empty results (e.g., from non-rank-0 processes in GRPO training)
        if not results:
            warnings.warn(
                f"Empty lm-eval results (likely from non-rank-0 process in GRPO training). "
                f"Returning 0.0 as placeholder."
            )
            return 0.0
        
        if metric_spec.dataset not in results:
            raise ValueError(
                f"Dataset '{metric_spec.dataset}' not found in lm-eval results. "
                f"Available datasets: {list(results.keys())}"
            )
        
        dataset_results = results[metric_spec.dataset]
        if metric_spec.metric_name not in dataset_results:
            raise ValueError(
                f"Metric '{metric_spec.metric_name}' not found in dataset '{metric_spec.dataset}'. "
                f"Available metrics: {list(dataset_results.keys())}"
            )
        
        value = dataset_results[metric_spec.metric_name]
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Metric value is not numeric: {value} (type: {type(value)})"
            )
        
        return float(value)
    
    # Case 2: HF Trainer format
    # Try with common prefixes
    prefixes = ["", "eval_", "test_", "valid_"]
    for prefix in prefixes:
        full_name = f"{prefix}{metric_name}"
        if full_name in eval_output:
            value = eval_output[full_name]
            if isinstance(value, (int, float)):
                return float(value)
    
    raise ValueError(
        f"Could not extract metric '{metric_name}' from evaluation output. "
        f"Available keys: {list(eval_output.keys())}"
    )


def extract_loss(eval_output: Dict[str, Any]) -> float:
    """
    Extract ONLY the loss metric from evaluation output.
    
    This is the PRIMARY function used by all attribution methods.
    
    Args:
        eval_output: Raw trainer.evaluate() output
    
    Returns:
        Loss value for attribution scoring
    
    Raises:
        ValueError: If loss cannot be found in output
    """
    return extract_metric(eval_output, metric_spec=None, metric_name="loss")


def align_and_average_metrics(
    eval_output: Dict[str, Any],
    metric_specs: List[LMEvalMetricSpec],
) -> float:
    """
    Extract multiple metrics, align them to the same direction (higher is better),
    and return their average.
    
    This is used by data models to handle multiple evaluation metrics consistently.
    
    Args:
        eval_output: Raw evaluation output (can be HF or lm-eval format)
        metric_specs: List of metric specifications
    
    Returns:
        Average of aligned metrics (all normalized to "higher is better")
    
    Raises:
        ValueError: If any metric cannot be extracted
    """
    if not metric_specs:
        raise ValueError("At least one metric_spec must be provided")
    
    aligned_values = []
    
    for spec in metric_specs:
        value = extract_metric(eval_output, metric_spec=spec)
        
        # Align: if lower is better, negate the value
        if not spec.higher_is_better:
            value = -value
        
        aligned_values.append(value)
    
    # Return the mean of aligned metrics
    return float(np.mean(aligned_values))
