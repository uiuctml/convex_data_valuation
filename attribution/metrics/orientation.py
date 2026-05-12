"""
Metric orientation registry (higher/lower is better).

This is the ONLY hardcoded registry - used for baseline comparison
in compare_baselines.py and run_from_scores.py.
"""
from typing import Dict


# Global orientation map: +1 = higher is better, -1 = lower is better
METRIC_ORIENTATION: Dict[str, int] = {
    # Loss-based metrics (lower is better)
    "loss": -1,
    "eval_loss": -1,
    "perplexity": -1,
    "ppl": -1,  # Common abbreviation for perplexity
    "cross_entropy": -1,
    "entropy": -1,  # Typically minimized for focused distributions
    
    # Accuracy-based metrics (higher is better)
    "accuracy": 1,
    "eval_accuracy": 1,
    "acc": 1,
    "exact_match": 1,
    "em": 1,
    "f1": 1,
    "token_f1": 1,  # Token-level F1 score
    "mean_token_accuracy": 1,  # Token-level accuracy
    
    # Reward-based metrics (higher is better)
    "reward": 1,
    "eval_reward": 1,
    "mean_reward": 1,
    
    # Code generation metrics (higher is better)
    "pass@1": 1,
    "pass@5": 1,
    "pass@10": 1,
    "pass@k": 1,
    
    # NLP metrics (higher is better)
    "bleu": 1,
    "rouge": 1,
    "rouge1": 1,
    "rouge2": 1,
    "rougeL": 1,
    "meteor": 1,
    "bertscore": 1,
}


def get_metric_orientation(metric_name: str) -> int:
    """
    Get orientation for a metric (+1 = higher better, -1 = lower better).
    
    Falls back to +1 (higher is better) if unknown.
    
    Args:
        metric_name: Metric name (can be with or without "eval_" prefix)
    
    Returns:
        +1 if higher is better, -1 if lower is better
    """
    # Try exact match first
    if metric_name in METRIC_ORIENTATION:
        return METRIC_ORIENTATION[metric_name]
    
    # Try normalized name (strip prefixes)
    name = metric_name.lower()
    for prefix in ["eval_", "test_", "valid_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    
    # Strip lm-eval suffixes (e.g., ",none", ",flexible-extract", ",strict-match")
    if "," in name:
        name = name.split(",")[0]
    
    # Strip _stderr suffix (these are not metrics to rank, but uncertainty measures)
    if name.endswith("_stderr"):
        # stderr metrics shouldn't be ranked, but if asked, treat same as base metric
        name = name[:-len("_stderr")]
    
    if name in METRIC_ORIENTATION:
        return METRIC_ORIENTATION[name]
    
    # Default: higher is better
    return 1
