from typing import Dict, List, Optional

def make_mix_weights(
    tasks: List[str], 
    target: str, 
    target_w: float, 
    aux_total_w: float,
    mode: str = "uniform",
    task_sizes: Optional[Dict[str, int]] = None,
    popularity_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute mixture weights for multitask training.
    
    Args:
        tasks: List of task names
        target: Target task name
        target_w: Weight for target task (used in uniform mode)
        aux_total_w: Total weight for auxiliary tasks (used in uniform mode)
        mode: Weighting strategy:
            - "uniform": Target gets target_w, aux tasks split aux_total_w evenly
            - "proportional": Weight by dataset size (requires task_sizes)
            - "popularity": Weight by custom popularity scores (requires popularity_weights)
        task_sizes: Dict mapping task -> number of examples (for proportional mode)
        popularity_weights: Dict mapping task -> popularity weight (for popularity mode)
    
    Returns:
        Dict mapping task -> normalized weight (sums to 1.0)
    """
    if (mode is None) or (mode == "uniform"):
        # Original logic: target gets target_w, aux tasks split aux_total_w evenly
        mix = {target: target_w}
        aux = [t for t in tasks if t != target]
        per = aux_total_w / max(1, len(aux))
        for t in aux: 
            mix[t] = per
    
    elif mode == "proportional":
        # Weight by dataset size
        if task_sizes is None:
            raise ValueError("task_sizes must be provided for 'proportional' mode")
        mix = {}
        for t in tasks:
            if t not in task_sizes:
                raise ValueError(f"Task '{t}' not found in task_sizes")
            mix[t] = float(task_sizes[t])
    
    elif mode == "popularity":
        # Weight by popularity scores
        if popularity_weights is None:
            raise ValueError("popularity_weights must be provided for 'popularity' mode")
        mix = {}
        for t in tasks:
            if t not in popularity_weights:
                raise ValueError(f"Task '{t}' not found in popularity_weights")
            mix[t] = float(popularity_weights[t])
    
    else:
        raise ValueError(f"Unknown mix_mode: {mode}. Choose from 'uniform', 'proportional', 'popularity'")
    
    # Normalize to sum to 1.0
    s = sum(mix.values())
    return {k: v / s for k, v in mix.items()}