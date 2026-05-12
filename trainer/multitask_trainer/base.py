from __future__ import annotations
from typing import Dict, List, Optional, Any

from transformers import AutoTokenizer
from trainer.multitask_trainer.allocate_tokens import make_mix_weights
from data.base_module import TaskSpec


class MultiTaskTrainerBase:
    """
    Shared helpers for multitask trainers (SFT, GRPO, etc.).

    This class is intentionally *stateless* w.r.t. DataLoader construction:
    - It does NOT implement `get_train_dataloader`.
    - Subclasses are responsible for building their own DataLoader
      (so GRPO can plug into TRL's trajectory machinery, and SFT can
       use a plain HF Trainer-style loader).

    What this base provides:
      * tokenizer init
      * TaskSpec building with few-shot logic
      * data_module reuse / task selection
      * mixture weight computation
      * token-budget → planned max_steps
      * a helper to build a MultiTaskOnTheFlyDataset
    """

    # ---------------------------------------------------------------------
    # Tokenizer initialization
    # ---------------------------------------------------------------------
    @staticmethod
    def _init_tokenizer(cfg) -> Any:
        tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token

        if hasattr(cfg, "padding_side") and cfg.padding_side in ("left", "right"):
            tok.padding_side = cfg.padding_side
        
        # Set model_max_length from config if specified (crucial for formatters that use tokenizer.model_max_length)
        if hasattr(cfg, "max_seq_length"):
            tok.model_max_length = cfg.max_seq_length

        return tok

    # ---------------------------------------------------------------------
    # Few-shot task spec builder
    # ---------------------------------------------------------------------
    @staticmethod
    def _build_task_specs(cfg) -> List[TaskSpec]:
        """
        Build TaskSpec list with few-shot control.
        Expects cfg to expose:
          - cfg.tasks: list of task names
          - cfg.few_shot: optional dict {task_name: k}
          - cfg.k_shot: global k for non-target tasks
          - cfg.target_task: target task name
          - cfg.few_shot_seed: seed for k-shot sampling
        """
        specs: List[TaskSpec] = []
        few_overrides = getattr(cfg, "few_shot", None) or {}

        for t in cfg.tasks:
            if t in few_overrides:
                k = few_overrides[t]
            elif getattr(cfg, "k_shot", None) is not None and t != cfg.target_task:
                k = cfg.k_shot
            else:
                k = None
            specs.append(TaskSpec(name=t, k_shot=k, k_shot_seed=cfg.few_shot_seed))

        return specs

    # ---------------------------------------------------------------------
    # Data module reuse / caching
    # ---------------------------------------------------------------------
    def _init_or_reuse_data_module(self, data_module: Optional[Any]):
        """
        If a data_module is passed, snapshot its original loaders so that
        multiple calls with different task overrides remain consistent.
        """
        if data_module is None:
            return None

        dm = data_module
        if not hasattr(dm, "_original_train_loaders"):
            dm._original_train_loaders = dict(getattr(dm, "train_loaders", {}))
            dm._original_eval_loaders = dict(getattr(dm, "eval_loaders", {}))
        return dm

    # ---------------------------------------------------------------------
    # Task selection + filtering
    # ---------------------------------------------------------------------
    def _select_tasks_and_filter_loaders(
        self,
        dm,
        tasks_override: Optional[List[str]],
    ):
        """
        Returns (task_names, train_loaders, eval_loaders) after applying
        an optional tasks_override list.

        Assumes dm exposes:
            - train_loaders: Dict[str, DataLoader]
            - eval_loaders: Dict[str, DataLoader]   (may be empty)
            - tasks (optional): List[TaskSpec or str]
        """

        if not hasattr(dm, "train_loaders"):
            raise ValueError("data_module must expose `train_loaders: Dict[str, DataLoader]`")
        if not hasattr(dm, "eval_loaders"):
            dm.eval_loaders = {}

        # Cache original loaders if not already cached (for multiple override calls)
        if not hasattr(dm, "_original_train_loaders"):
            dm._original_train_loaders = dict(dm.train_loaders)
            dm._original_eval_loaders = dict(dm.eval_loaders)

        if tasks_override is None:
            if hasattr(dm, "tasks"):
                task_names = [ts.name if hasattr(ts, "name") else ts for ts in dm.tasks]
            else:
                task_names = sorted(dm._original_train_loaders.keys())
            # ensure current loaders line up
            dm.train_loaders = {
                t: dm._original_train_loaders[t]
                for t in task_names
                if t in dm._original_train_loaders
            }
            dm.eval_loaders = {
                t: dm._original_eval_loaders[t]
                for t in task_names
                if t in dm._original_eval_loaders
            }
        else:
            task_names = list(tasks_override)
            # Always restore from original cached loaders for consistent override behavior
            dm.train_loaders = {
                t: dm._original_train_loaders[t]
                for t in task_names
                if t in dm._original_train_loaders
            }
            dm.eval_loaders = {
                t: dm._original_eval_loaders[t]
                for t in task_names
                if t in dm._original_eval_loaders
            }

        return task_names, dm.train_loaders, dm.eval_loaders

    # ---------------------------------------------------------------------
    # Max steps planning
    # ---------------------------------------------------------------------
    def _plan_max_steps(
        self,
        cfg,
        task_names: List[str],
        mix: Dict[str, float],
    ):
        """
        Return planned max_steps from cfg.max_steps and per-task step allocation.
        """
        planned_max_steps = max(1, int(cfg.max_steps))

        per_task_steps: Dict[str, int] = {}
        for t in task_names:
            per_task_steps[t] = int(planned_max_steps * mix[t])

        return planned_max_steps, per_task_steps

    # ---------------------------------------------------------------------
    # Mixture weights
    # ---------------------------------------------------------------------
    def _compute_mix(
        self,
        cfg,
        task_names: List[str],
        mix_override: Optional[Dict[str, float]],
        train_loaders: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute per-task mixture weights, caching the *original* target_weight
        and aux_weight_total on the cfg for reuse across multiple calls.
        
        Supports three modes:
          - "uniform": target_weight + aux_weight_total split evenly
          - "proportional": weight by dataset size
          - "popularity": weight by custom popularity scores
        """
        if (
            not hasattr(cfg, "_original_target_weight")
            or not hasattr(cfg, "_original_aux_weight_total")
            or cfg._original_aux_weight_total is None
        ):
            cfg._original_target_weight = getattr(cfg, "target_weight", 1.0)
            cfg._original_aux_weight_total = getattr(cfg, "aux_weight_total", 0.0)

        if mix_override is None:
            mode = getattr(cfg, "mix_mode", "uniform")
            
            # Extract dataset sizes for proportional mode
            task_sizes = None
            if mode == "proportional":
                if train_loaders is None:
                    raise ValueError("train_loaders must be provided for 'proportional' mix_mode")
                task_sizes = {}
                for t in task_names:
                    if t in train_loaders:
                        loader = train_loaders[t]
                        # Try to get dataset size from loader.dataset
                        if hasattr(loader, "dataset") and hasattr(loader.dataset, "__len__"):
                            task_sizes[t] = len(loader.dataset)
                        else:
                            raise ValueError(f"Cannot determine size of dataset for task '{t}'")
                    else:
                        raise ValueError(f"Task '{t}' not found in train_loaders")
            
            # Extract popularity weights for popularity mode
            popularity_weights = None
            if mode == "popularity":
                popularity_weights = getattr(cfg, "popularity_weights", None)
                if popularity_weights is None:
                    raise ValueError("popularity_weights must be specified in config for 'popularity' mix_mode")
            
            return make_mix_weights(
                task_names,
                cfg.target_task,
                cfg._original_target_weight,
                cfg._original_aux_weight_total,
                mode=mode,
                task_sizes=task_sizes,
                popularity_weights=popularity_weights,
            )
        return dict(mix_override)

    # ---------------------------------------------------------------------
    # Eval dataloader hook (optional helper)
    # ---------------------------------------------------------------------
    def get_eval_dataloader(self, eval_dataset: Optional[Any] = None):
        """
        Default eval behavior: reuse the target-task eval loader if available,
        otherwise defer to the parent Trainer implementation.

        Subclasses must:
            - set self._eval_loaders : Dict[str, DataLoader]
            - set self.cfg.target_task : str
        """

        if eval_dataset is not None:
            return super().get_eval_dataloader(eval_dataset)

        t = getattr(self.cfg, "target_task", None)
        if t is None:
            raise ValueError("Expected cfg.target_task to be defined.")

        if not hasattr(self, "_eval_loaders") or t not in self._eval_loaders:
            print(f"[MultiTaskBase] No eval loader found for {t}, deferring to base.")
            return super().get_eval_dataloader(eval_dataset)

        return self._eval_loaders[t]