# multisource_data/attribution/datamodel_uniform.py
from __future__ import annotations
from typing import Dict, List
import numpy as np
import tempfile
from attribution.datamodel.base import DataModelAttributionBase, DataModelFitCfg, _row_rng



# ---------------- Concrete Implementation ----------------

class UniformSamplingDataModel(DataModelAttributionBase):
    """
    Data model where each row includes a uniform random subset of auxiliaries
    of size round(include_fraction * N_aux). The target column is always 1.

    For each row:
      - tasks = [target] + selected_aux
      - build a row-specific cfg (copy of mtl_cfg with tasks restricted)
      - HFMultiTaskTrainer(row_cfg).train()   # steps planned from row's token budget internally
      - eval on target: collect loss from metrics, accuracy from predictions
    """

    # --- Design matrix (uniform sampling) ---
    def _sample_row(self, *, n_aux: int, seed: int, row_idx: int, fit_cfg: DataModelFitCfg) -> np.ndarray:
        g = _row_rng(seed, row_idx)
        k = max(1, int(round(float(fit_cfg.include_fraction) * n_aux))) if n_aux > 0 else 0
        row = np.zeros(1 + n_aux, dtype=np.float32)
        row[0] = 1.0
        if k > 0:
            idx = g.choice(n_aux, size=k, replace=False)
            row[1 + idx] = 1.0
        return row

    def measure_row(
        self,
        *,
        selection_row: np.ndarray,
        aux_names: List[str],
    ) -> Dict[str, float]:
        # Get target task from trainer_factory
        target_task_name = self.factory.target_task
        
        # pick auxiliaries for this row
        selected_aux = [aux_names[j] for j in range(len(aux_names)) if selection_row[1 + j] == 1.0]
        row_tasks = [target_task_name] + selected_aux

        with tempfile.TemporaryDirectory(prefix="uniform_row_") as tmpdir:
            trainer = self.factory.make_row_trainer(
                tasks=row_tasks,
                output_dir=tmpdir,
            )
            trainer.train()
            raw_metrics = trainer.evaluate()
        
        # Extract metrics (either loss only or multiple lm-eval metrics)
        # The factory will handle alignment and averaging if multiple metrics specified
        metric_value = getattr(self, 'metric_extractor_fn', None)
        if metric_value is None:
            # Default: extract loss only
            metric_value = self.factory.extract_loss_from_eval(raw_metrics)
        else:
            # Use custom extractor (set by score_auxiliary_datasets)
            metric_value = metric_value(raw_metrics)
        
        return {
            "metric": metric_value,
        }