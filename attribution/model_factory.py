from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import copy
import torch
from torch import nn
from trainer.trainer_factory import make_trainer
from modeling.params_factory import get_param_strategy
from attribution.metrics.extractor import (
    extract_loss,
    align_and_average_metrics,
    LMEvalMetricSpec,
)



@dataclass
class TrainerModelFactory:
    """
    Adapter around a (multi-task) Trainer that exposes a simple interface
    for attribution methods:

      - model_init_fn(): returns a model whose parameters are reset to a
        fixed base state θ₀ (typically the initial trainer.model params)
      - loss_fn(model, batch): computes loss using trainer.compute_loss
      - target_loader: train loader for cfg.target_task
      - aux_loaders:  {aux_task: train_loader}
      - data_module:  underlying data module (dm)

    Additionally exposes trainer-related infrastructure needed by old
    AttributionDataLoader logic:
      - tokenizer  (trainer.tok or trainer.tokenizer)
      - collator   (trainer.data_collator)
      - model_name, datadir, max_seq_length, num_workers, pin_memory,
        drop_last, train_batch_size, eval_batch_size, few_shot_seed, k_shot
    """

    trainer: Any
    target_task: Optional[str] = None
    aux_tasks: Optional[List[str]] = None

    # Populated in __post_init__
    data_module: Any = field(init=False)
    target_loader: Any = field(init=False)
    aux_loaders: Dict[str, Any] = field(init=False)
    trainable_param_filter: Optional[Callable[[str, nn.Parameter], bool]] = field(
        init=False, default=None
    )

    # Exposed infra
    tokenizer: Any = field(init=False, default=None)
    collator: Any = field(init=False, default=None)

    # Common config-derived attributes
    model_name: Optional[str] = field(init=False, default=None)
    datadir: Optional[str] = field(init=False, default=None)
    max_seq_length: int = field(init=False, default=512)
    num_workers: int = field(init=False, default=4)
    pin_memory: bool = field(init=False, default=True)
    drop_last: bool = field(init=False, default=False)
    train_batch_size: int = field(init=False, default=16)  # unique prompts per batch (for GRPO: per_device_bs / num_gen)
    eval_batch_size: int = field(init=False, default=64)
    num_generations: int = field(init=False, default=1)  # GRPO num_generations (1 for non-GRPO)
    few_shot_seed: int = field(init=False, default=42)
    k_shot: Optional[int] = field(init=False, default=None)

    # CPU snapshot of base parameters θ₀
    _base_state: Optional[Dict[str, torch.Tensor]] = field(init=False, default=None)

    def __post_init__(self):
        # ---------------- cfg & basic checks ----------------
        cfg = getattr(self.trainer, "cfg", None)
        if cfg is None:
            raise ValueError("TrainerModelFactory expects trainer to have a `cfg` attribute.")

        # ---------------- target / aux tasks ----------------
        if self.target_task is None:
            self.target_task = getattr(cfg, "target_task", None)
        if self.target_task is None:
            raise ValueError("TrainerModelFactory could not infer `target_task` from cfg or arguments.")

        if self.aux_tasks is None:
            task_names = getattr(self.trainer, "task_names", None)
            if task_names is None:
                # fallback: derive from data module loaders
                dm = getattr(self.trainer, "dm", None)
                if dm is None or not hasattr(dm, "train_loaders"):
                    raise ValueError(
                        "TrainerModelFactory could not infer task names: "
                        "trainer.task_names missing and trainer.dm.train_loaders not available."
                    )
                task_names = list(dm.train_loaders.keys())
            self.aux_tasks = [t for t in task_names if t != self.target_task]

        # ---------------- data module & loaders ----------------
        dm = getattr(self.trainer, "dm", None)
        if dm is None or not hasattr(dm, "train_loaders"):
            raise ValueError("TrainerModelFactory expects trainer.dm with `train_loaders`.")
        self.data_module = dm

        if self.target_task not in dm.train_loaders:
            raise KeyError(f"Target task '{self.target_task}' not found in dm.train_loaders.")

        self.target_loader = dm.train_loaders[self.target_task]
        self.aux_loaders = {
            t: dm.train_loaders[t]
            for t in self.aux_tasks
            if t in dm.train_loaders
        }

        # ---------------- tokenizer / collator ----------------
        tok = getattr(self.trainer, "tok", None)
        if tok is None:
            tok = getattr(self.trainer, "tokenizer", None)
        self.tokenizer = tok

        self.collator = getattr(self.trainer, "data_collator", None)

        # ---------------- cfg-derived attributes ----------------
        self.model_name = getattr(cfg, "model_name", None)
        self.datadir = getattr(cfg, "datadir", None)

        self.num_workers = int(getattr(cfg, "num_workers", 4))
        self.pin_memory = bool(getattr(cfg, "pin_memory", True))
        self.drop_last = bool(getattr(cfg, "drop_last", False))
        
        trainer_type = getattr(cfg, "trainer_type", "sft")
        num_generations = getattr(cfg, "num_generations", 1) or 1
        per_device_bs = getattr(cfg, "per_device_train_batch_size", None) or getattr(cfg, "train_batch_size", 16)
        
        if trainer_type == "grpo" and num_generations > 1:
            # For GRPO: unique prompts per batch = per_device_train_batch_size / num_generations
            self.train_batch_size = int(per_device_bs // num_generations)
        else:
            self.train_batch_size = int(per_device_bs)
        
        self.num_generations = num_generations  # Store for reference
        self.eval_batch_size = int(getattr(cfg, "eval_batch_size", None) or 
                                    getattr(cfg, "per_device_eval_batch_size", 16))
        self.few_shot_seed = int(getattr(cfg, "few_shot_seed", 42))
        self.k_shot = getattr(cfg, "k_shot", None)

        # ---------------- trainable_param_filter ----------------
        trainer_type = getattr(cfg, "trainer_type", "sft")
        param_strategy = get_param_strategy(trainer_type, cfg)
        self.trainable_param_filter = param_strategy.make_filter()

        # ---------------- snapshot base parameters θ₀ ----------------
        # We do this once, on CPU, from the *unwrapped* model if accelerate is used.
        accelerator = getattr(self.trainer, "accelerator", None)
        model = self.trainer.model
        if accelerator is not None and hasattr(accelerator, "unwrap_model"):
            model = accelerator.unwrap_model(model)

        # state_dict() is already on appropriate devices; we move to CPU + clone
        self._base_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    # ------------------------------------------------------------------
    # Parameter reset helper
    # ------------------------------------------------------------------
    def reset_trainer_model_to_base(self) -> None:
        """
        Reset trainer.model's parameters back to the stored base_state θ₀.
        Works uniformly for single-GPU and multi-GPU (DDP/accelerate) setups.
        """
        if self._base_state is None:
            raise RuntimeError("No base_state snapshot stored in TrainerModelFactory.")

        accelerator = getattr(self.trainer, "accelerator", None)
        model = self.trainer.model
        if accelerator is not None and hasattr(accelerator, "unwrap_model"):
            raw_model = accelerator.unwrap_model(model)
        else:
            raw_model = model

        # Load CPU snapshot back into the model; strict=False allows minor head differences if any.
        raw_model.load_state_dict(self._base_state, strict=False)

    # ------------------------------------------------------------------
    # Model init and loss fn
    # ------------------------------------------------------------------
    @property
    def model_init_fn(self) -> Callable[[], Any]:
        """
        Returns a function that, when called, resets `trainer.model`'s
        parameters to the base snapshot θ₀ and returns that model.

        This is consistent for all models (small/large, single/multi-GPU)
        and avoids creating extra full copies of the model in memory.
        """
        def _init():
            self.reset_trainer_model_to_base()
            return self.trainer.model

        return _init

    def loss_fn(self, model: Any, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for gradient-based attribution.
        
        For GRPO: batch is list[dict] from collator. We need to:
        1. Duplicate each prompt num_generations times (as TRL's RepeatSampler does)
        2. Generate completions from prompts
        3. Compute rewards
        4. Build proper GRPO batch dict with advantages
        5. Call trainer.compute_loss
        
        For MLM/others: Delegates to trainer.compute_loss directly.
        """
        trainer_type = getattr(self.trainer.cfg, "trainer_type", "sft")
        
        if trainer_type == "grpo":
            # For GRPO, batch is list[dict] from GRPOCollator
            # We need to convert it to proper GRPO format using trainer's _prepare_inputs
            # which handles generation, reward computation, and advantage calculation
            
            # CRITICAL: TRL's GRPO expects batches to have each prompt duplicated 
            # num_generations times. In normal training, RepeatSampler does this.
            # We must replicate this behavior for our attribution batches.
            num_generations = self.trainer.num_generations
            
            # Duplicate each sample num_generations times (same as RepeatSampler)
            # batch is list[dict], we repeat each dict num_generations times
            duplicated_batch = []
            for sample in batch:
                for _ in range(num_generations):
                    duplicated_batch.append(sample)
            
            # Move model to eval mode for generation, then back to train for loss
            was_training = model.training
            model.eval()
            
            # Use trainer's _prepare_inputs which does generation + reward + advantages
            with torch.no_grad():
                prepared_batch = self.trainer._prepare_inputs(duplicated_batch)
            
            # Back to training mode for loss computation
            if was_training:
                model.train()
            
            # Now compute loss with the properly prepared batch
            return self.trainer._compute_loss(model, prepared_batch)
        else:
            # For SFT and other trainers, use their compute_loss
            # HF Trainer `compute_loss` signature:
            #   compute_loss(model, inputs, return_outputs=False)
            return self.trainer.compute_loss(model, batch, return_outputs=False)

    def extract_loss_from_eval(self, eval_output: Dict[str, Any]) -> float:
        """
        Extract ONLY the loss metric from trainer.evaluate() output.
        
        This is the PRIMARY method for all attribution methods - they should
        ONLY use loss for scoring, never other metrics.
        
        Args:
            eval_output: Raw output from trainer.evaluate()
        
        Returns:
            Loss value for attribution scoring
        """
        return extract_loss(eval_output)
    
    def extract_metrics_from_eval(
        self,
        eval_output: Dict[str, Any],
        metric_specs: Optional[List[LMEvalMetricSpec]] = None,
    ) -> float:
        """
        Extract and aggregate metrics from trainer.evaluate() output.
        
        Supports both HF trainer format and lm-eval format:
        - If metric_specs is None: extracts loss only (HF format)
        - If metric_specs provided: extracts specified metrics, aligns them
          to same direction (higher is better), and returns their average
        
        This is used by data model methods to support multiple metrics.
        
        Args:
            eval_output: Raw output from trainer.evaluate() or lm-eval
            metric_specs: List of LMEvalMetricSpec for lm-eval format (None = use loss)
        
        Returns:
            Metric value (or average of multiple aligned metrics)
        """
        if metric_specs is None:
            # Default: use loss only (HF format)
            return extract_loss(eval_output)
        else:
            # lm-eval format: extract multiple metrics, align, and average
            return align_and_average_metrics(eval_output, metric_specs)

    # ------------------------------------------------------------------
    # Row-specific trainer for datamodel methods
    # ------------------------------------------------------------------
    def make_row_trainer(
        self,
        tasks: List[str],
        output_dir: str,
    ) -> Any:
        """
        Build a trainer for a single row of a data-model experiment.

        Reuses:
          - self.trainer.cfg as the base config (includes k-shot changes)
          - self.data_module so we don't reload/reformat datasets

        Args:
            tasks:        [target] + selected auxiliaries for this row.
            output_dir:   where this row's run artifacts should go.

        Returns:
            A new trainer instance configured for this row.
        """
        local_cfg = copy.deepcopy(self.trainer.cfg)
        local_cfg.tasks = list(tasks)

        return make_trainer(
            cfg=local_cfg,
            tasks_override=tasks,
            output_dir_override=output_dir,
            data_module=self.data_module,
        )
    # ------------------------------------------------------------------
    # Single task finetuning trainer for task vectors
    # ------------------------------------------------------------------
    def make_single_task_trainer(
        self,
        task_name: str,
        *,
        output_dir: str,
        mix_override: Optional[Dict[str, float]] = None,
    ) -> Any:
        """
        Build a trainer for a single task, optionally overriding
        the global token budget or mixture weights.

        Reuses:
          - self.trainer.cfg  (same optimizer, schedule, model setup)
          - self.data_module  (no dataset rebuilding)
          - self.trainer_type logic from make_trainer

        Args:
            task_name:
                Name of the task to fine-tune on.

            output_dir:
                Directory where this run's outputs and checkpoints go.

            mix_override:
                Optional explicit mix weights dict. Defaults to {task_name: 1.0}.

        Returns:
            A fresh trainer instance configured for this single task.
        """
        local_cfg = copy.deepcopy(self.trainer.cfg)

        # Focus only on this one task
        local_cfg.tasks = [task_name]

        # Ensure trainer thinks this is the target (important for evaluation)
        if hasattr(local_cfg, "target_task"):
            local_cfg.target_task = task_name

        # Default mixture = 1.0 for this single task
        if mix_override is None:
            mix_override = {task_name: 1.0}

        # Reuse pre-loaded data module to avoid rebuilding datasets
        trainer = make_trainer(
            cfg=local_cfg,
            tasks_override=[task_name],
            mix_override=mix_override,
            output_dir_override=output_dir,
            data_module=self.data_module,
        )

        return trainer

def make_aux_kshot_model_factory(
    cfg: Any,
    aux_k_shot: int,
    debug_mode: bool = False,
) -> TrainerModelFactory:
    """
    Build a Trainer + TrainerModelFactory where:

      - target task uses FULL data (k_shot=None) in normal mode
      - target task uses k_shot = aux_k_shot in DEBUG mode (for faster testing)
      - every auxiliary task uses k_shot = aux_k_shot

    Args:
        cfg: Base configuration
        aux_k_shot: Number of shots for auxiliary tasks (and target if debug_mode=True)
        debug_mode: If True, also apply k_shot to target task for faster debugging

    Implementation details:
      - We disable cfg.k_shot globally (so the generic TaskSpec logic
        won't touch non-target tasks with a global k).
      - We then set cfg.few_shot[aux_task] = aux_k_shot for all aux tasks,
        where aux tasks are all tasks except cfg.target_task.
      - In debug_mode, we also set cfg.few_shot[target_task] = aux_k_shot.
    """
    local_cfg = copy.deepcopy(cfg)

    # 1) Identify aux tasks
    all_tasks = list(local_cfg.tasks)
    target = local_cfg.target_task
    aux_tasks = [t for t in all_tasks if t != target]

    # 2) Disable global k_shot
    if hasattr(local_cfg, "k_shot"):
        local_cfg.k_shot = None

    # 3) Per-aux-task k-shot via few_shot
    few = getattr(local_cfg, "few_shot", None) or {}
    # Overwrite / set k for each auxiliary
    for t in aux_tasks:
        few[t] = int(aux_k_shot)
    
    # DEBUG: Also apply k-shot to target task for faster testing
    if debug_mode:
        few[target] = int(aux_k_shot)
    
    local_cfg.few_shot = few

    # Now MultiTaskTrainerBase._build_task_specs will:
    #   - see few_shot[aux_task] = aux_k_shot → k = aux_k_shot
    #   - see k_shot=None → no global k
    #   - for target_task, no entry in few_shot → k = None (full data)

    trainer = make_trainer(
        cfg=local_cfg,
        # no tasks_override / data_module reuse here; this is a clean entrypoint
    )

    factory = TrainerModelFactory(trainer=trainer)
    return factory


def make_full_data_model_factory(
    cfg: Any,
) -> TrainerModelFactory:
    """
    Build a Trainer + TrainerModelFactory where *all tasks use full data*
    (no k-shot), regardless of what's in cfg.

    This is useful for:
      - final full-dataset training
      - attribution methods that conceptually assume you train on the
        entire dataset rather than k-shot subsets.
    """
    # Shallow/deep copy cfg so we don't mutate the original
    local_cfg = copy.deepcopy(cfg)

    # 1) Disable global k-shot
    if hasattr(local_cfg, "k_shot"):
        local_cfg.k_shot = None

    # 2) Disable per-task few_shot overrides
    if hasattr(local_cfg, "few_shot"):
        # could also delattr(local_cfg, "few_shot"), but empty dict is explicit
        local_cfg.few_shot = {}

    # 3) Build trainer exactly as usual, but now TaskSpec will have k_shot=None
    #    for all tasks because both k_shot and few_shot are disabled.
    trainer = make_trainer(
        cfg=local_cfg,
        # Again, no tasks_override / data_module reuse here for this entrypoint.
    )

    factory = TrainerModelFactory(trainer=trainer)
    return factory
