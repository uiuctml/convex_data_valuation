"""
Utility functions for parsing metric configurations.
"""
from typing import Optional, List


def parse_metric_specs_from_config(cfg) -> Optional[List]:
    """
    Parse metric specifications from config.
    
    Config format:
        metric_specs:
          - dataset: gsm8k
            metric_name: exact_match,flexible-extract
            higher_is_better: true
          - dataset: hendrycks_math
            metric_name: exact_match,strict-match
            higher_is_better: true
    
    Returns:
        List of LMEvalMetricSpec or None if not specified
    
    Note: Import LMEvalMetricSpec inside function to avoid circular imports
    """
    if not hasattr(cfg, 'metric_specs') or cfg.metric_specs is None:
        return None
    
    # Import here to avoid circular dependency
    from attribution.metrics.extractor import LMEvalMetricSpec
    
    specs = []
    for spec_dict in cfg.metric_specs:
        specs.append(LMEvalMetricSpec(
            dataset=spec_dict['dataset'],
            metric_name=spec_dict['metric_name'],
            higher_is_better=spec_dict.get('higher_is_better', True),
        ))
    
    return specs
