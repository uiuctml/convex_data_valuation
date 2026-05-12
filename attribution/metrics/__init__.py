"""
Metric utilities for attribution methods.

Key principles:
- Attribution methods can use loss or lm-eval metrics (with metric specs)
- Metric discovery is used only in run_from_scores.py for final evaluation
- Metric extraction handles trainer-specific output formats and lm-eval format
"""

from .inference import (
    infer_available_metrics,
    get_attribution_metric,
)
from .config_utils import (
    parse_metric_specs_from_config,
)
from .extractor import (
    MetricExtractor,
    LMEvalMetricSpec,
    extract_metric,
    extract_loss,
    align_and_average_metrics,
)
from .orientation import (
    get_metric_orientation,
    METRIC_ORIENTATION,
)

__all__ = [
    "infer_available_metrics",
    "get_attribution_metric",
    "parse_metric_specs_from_config",
    "MetricExtractor",
    "LMEvalMetricSpec",
    "extract_metric",
    "extract_loss",
    "align_and_average_metrics",
    "get_metric_orientation",
    "METRIC_ORIENTATION",
]
