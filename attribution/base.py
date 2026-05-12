from __future__ import annotations

"""
Base abstract class for all dataset-level attribution strategies.
This should remain minimal. Specialized subclasses (e.g., gradient-based, data-model-based)
will extend this base in separate files.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict
import torch

if TYPE_CHECKING:
    from attribution.model_factory import TrainerModelFactory


class DataAttribution(ABC):
    """Abstract base for dataset-level attribution strategies.

    All attribution methods now use a TrainerModelFactory that provides:
      - model_init_fn: fresh model initialization
      - trainable_param_filter: which params to tune
      - device: computation device
      - target/aux loaders, data_module, etc.

    Subclasses should implement `score_auxiliary_datasets`, returning a mapping from
    dataset name to scalar attribution score.
    """

    def __init__(
        self,
        trainer_factory: "TrainerModelFactory",
    ):
        if trainer_factory is None:
            raise ValueError("DataAttribution requires a TrainerModelFactory instance.")
        
        self.factory = trainer_factory
        
        # Extract commonly-used attributes from factory for convenience
        self.model_init_fn = trainer_factory.model_init_fn
        self.trainable_param_filter = trainer_factory.trainable_param_filter
        
        # Get device from trainer.args.device (HF Trainer standard)
        trainer = trainer_factory.trainer
        if hasattr(trainer, "args") and hasattr(trainer.args, "device"):
            self.device = trainer.args.device
        else:
            # Fallback if args.device is not available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def score_auxiliary_datasets(self, *args, **kwargs) -> Dict[str, float]:
        """Compute attribution scores for auxiliary datasets.

        Implementations are free to choose their own argument signatures, but should
        return a dict mapping dataset_name -> float score.
        """
        raise NotImplementedError