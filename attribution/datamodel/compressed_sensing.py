from __future__ import annotations

import math
import tempfile
from typing import Dict, List

import numpy as np

from attribution.datamodel.base import (
    DataModelAttributionBase,
    DataModelFitCfg,
    _row_rng,
)


class CompressedSensingDataModel(DataModelAttributionBase):
    """
    Compressed-sensing data model.

    For aux columns (n = #aux):
      a_{i,j}^{CS} ~ sqrt(3/n) * { +1 w.p. 1/6, 0 w.p. 2/3, -1 w.p. 1/6 }
    Target column is fixed to 1 (not scaled, not randomized).

    For each row i:
      - Build S1, S2 from signs (+1 in S1 \\ S2, -1 in S2 \\ S1, 0 in intersection)
      - Train two models on [target] + S1 and [target] + S2
      - Measure metrics on target eval
      - y_i^{CS} = sqrt(3/n) * ( metric(S1) - metric(S2) ) for each metric
    """

    # ------------- Design matrix sampler -------------
    def _sample_row(self, *, n_aux: int, seed: int, row_idx: int, fit_cfg: DataModelFitCfg) -> np.ndarray:
        row = np.zeros(1 + n_aux, dtype=np.float32)
        row[0] = 1.0  # target fixed
        if n_aux == 0:
            return row
        g = _row_rng(seed, row_idx)
        scale = math.sqrt(3.0 / float(n_aux))
        vals  = np.array([+scale, 0.0, -scale], dtype=np.float32)
        probs = np.array([1.0/6.0, 2.0/3.0, 1.0/6.0], dtype=np.float32)
        row[1:] = g.choice(vals, size=n_aux, p=probs)
        return row

    # ------------- Per-row measurement -------------
    def measure_row(
        self,
        *,
        selection_row: np.ndarray,       # one row of A (length 1 + n_aux); aux entries are scaled signs
        aux_names: List[str],
    ) -> Dict[str, float]:
        """
        Build S1, S2 from the (scaled) signs in selection_row and return
        rescaled metric differences y_i^{CS} for each metric:
            acc:  sqrt(3/n) * (acc1 - acc2)
            loss: sqrt(3/n) * (loss1 - loss2)
        """
        # Get target task from trainer_factory
        target_task_name = self.factory.target_task
        
        n_aux = len(aux_names)
        if n_aux == 0:
            # no aux: both S1 and S2 are empty; difference is 0
            return {"metric": 0.0}

        # Recover signs in {-1,0,+1} from scaled entries (tolerant to floating error)
        scale = math.sqrt(3.0 / float(n_aux))
        raw = selection_row[1:]  # aux part
        # Map close-to-zero to 0, positive to +1, negative to -1
        eps = 1e-8
        signs = np.zeros_like(raw, dtype=np.int8)
        signs[raw >= +scale - eps] = +1
        signs[raw <= -scale + eps] = -1
        # Remaining values close to 0 stay 0

        # Build S1, S2 per definition:
        #   +1 -> in S1 \ S2
        #   -1 -> in S2 \ S1
        #    0 -> in S1 ∩ S2
        S1 = [aux_names[j] for j in range(n_aux) if signs[j] >= 0]  # 0 and +1
        S2 = [aux_names[j] for j in range(n_aux) if signs[j] <= 0]  # 0 and -1

        # Train two models (S1 vs S2), evaluate on target
        # Use temp dirs to avoid disk bloat
        def _train_eval(aux_subset: List[str]) -> float:
            row_tasks = [target_task_name] + aux_subset

            with tempfile.TemporaryDirectory(prefix=f"cs_row_") as tmpdir:
                # Reuse prebuilt TrainerModelFactory (k-shot / full-data logic, shared DM)
                trainer = self.factory.make_row_trainer(
                    tasks=row_tasks,
                    output_dir=tmpdir,
                )
                trainer.train()
                raw_metrics = trainer.evaluate()

            # Extract metrics (either loss only or multiple lm-eval metrics)
            metric_extractor_fn = getattr(self, 'metric_extractor_fn', None)
            if metric_extractor_fn is None:
                # Default: extract loss only
                metric_value = self.factory.extract_loss_from_eval(raw_metrics)
            else:
                # Use custom extractor (set by score_auxiliary_datasets)
                metric_value = metric_extractor_fn(raw_metrics)
            return metric_value


        metric1 = _train_eval(S1)
        metric2 = _train_eval(S2)

        # Response vector y_i^{CS} is scaled difference
        y_metric = scale * (metric1 - metric2)
        return {"metric": y_metric}