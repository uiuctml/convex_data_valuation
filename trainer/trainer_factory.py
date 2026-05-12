from __future__ import annotations
import copy
import os
from datetime import datetime
from typing import Optional, List, Any, Dict

from transformers import set_seed
from trl import GRPOConfig, SFTConfig

from trainer.multitask_trainer.grpo.multitask_grpo_trainer import GRPOMultiTaskTrainer
from trainer.multitask_trainer.sft.multitask_sft_trainer import SFTMultiTaskTrainer


# ----------------------------------------------------------------------
# TOP-LEVEL DISPATCH
# ----------------------------------------------------------------------

def make_trainer(
    *,
    cfg: Any,
    tasks_override: Optional[List[str]] = None,
    mix_override: Optional[Dict[str, float]] = None,
    output_dir_override: Optional[str] = None,
    data_module: Optional[Any] = None,
    seed_offset: int = 0,
    **extra_kwargs,
):
    """
    Generic entry point.

    Automatically picks the right multitask trainer based on cfg.trainer_type:
      - "sft"  → SFTMultiTaskTrainer (instruction finetuning)
      - "grpo" → GRPOMultiTaskTrainer (RL style)

    Default: "sft" if not set in cfg.
    """
    trainer_type = getattr(cfg, "trainer_type", "sft").lower()

    if trainer_type == "grpo":
        return make_grpo_trainer(
            cfg=cfg,
            tasks_override=tasks_override,
            output_dir_override=output_dir_override,
            seed_offset=seed_offset,
            mix_override=mix_override,
            **extra_kwargs,
        )

    elif trainer_type == "sft":
        return make_sft_trainer(
            cfg=cfg,
            tasks_override=tasks_override,
            output_dir_override=output_dir_override,
            seed_offset=seed_offset,
            mix_override=mix_override,
            data_module=data_module,
            **extra_kwargs,
        )

    else:
        raise ValueError(
            f"Unknown trainer_type '{trainer_type}'. Expected one of: ['grpo', 'sft']."
        )

# ----------------------------------------------------------------------
# GRPO MULTITASK TRAINER (MATH RL)
# ----------------------------------------------------------------------

def make_grpo_trainer(
    *,
    cfg: Any,
    tasks_override: Optional[List[str]] = None,
    mix_override: Optional[Dict[str, float]] = None,
    output_dir_override: Optional[str] = None,
    data_module: Optional[Any] = None,
    seed_offset: int = 0,
) -> GRPOMultiTaskTrainer:
    """
    Build a GRPOMultiTaskTrainer for multitask RL (e.g., MATH with GRPO).

    Important knobs routed from cfg:
      - standard training args (lr, weight_decay, warmup_ratio, etc.)
      - GRPO-specific: num_generations, max_prompt_length, max_completion_length,
        temperature, beta
      - vLLM: use_vllm, vllm_mode
      - stability: gradient_checkpointing, gradient_accumulation_steps,
        bf16/fp16, max_grad_norm
    """

    # ---- per-trainer seed ----
    base_seed = getattr(cfg, "seed", 42)
    training_seed = base_seed + seed_offset
    set_seed(training_seed)

    # ---- eval cadence ----
    if int(cfg.eval_every) == -1:
        eval_strategy = "no"
        eval_steps = None
    else:
        eval_strategy = "steps"
        eval_steps = cfg.eval_every

    # ---- shallow copy cfg if we override tasks ----
    local_cfg = cfg if tasks_override is None else copy.deepcopy(cfg)
    if tasks_override is not None:
        local_cfg.tasks = list(tasks_override)

    # ---- GRPOConfig construction ----
    # Standard training knobs
    learning_rate = getattr(cfg, "lr", 1e-5)
    weight_decay = getattr(cfg, "weight_decay", 0.0)
    warmup_ratio = getattr(cfg, "warmup_ratio", 0.0)
    lr_scheduler_type = getattr(cfg, "lr_scheduler_type", "constant")
    optim = getattr(cfg, "optim", "adamw_torch")
    max_steps = getattr(cfg, "max_steps", 60)
    num_train_epochs = getattr(cfg, "num_train_epochs", 1)

    # Batch / precision / stability
    per_device_train_batch_size = getattr(cfg, "per_device_train_batch_size", cfg.train_batch_size)
    # For eval batch size, provide fallback chain: per_device_eval_batch_size -> eval_batch_size -> train_batch_size
    per_device_eval_batch_size = getattr(cfg, "per_device_eval_batch_size", 
                                          getattr(cfg, "eval_batch_size", per_device_train_batch_size))
    grad_accum = getattr(cfg, "grad_accum_steps", 8)
    bf16 = getattr(cfg, "bf16", False)
    fp16 = getattr(cfg, "fp16", False)
    gradient_checkpointing = getattr(cfg, "gradient_checkpointing", False)
    gradient_checkpointing_kwargs = getattr(
        cfg,
        "gradient_checkpointing_kwargs",
        {"use_reentrant": False},
    )
    max_grad_norm = getattr(cfg, "max_grad_norm", 1.0)

    # GRPO / generation knobs
    num_generations = getattr(cfg, "num_generations", 8)
    max_prompt_length = getattr(cfg, "max_prompt_length", 1024)
    max_completion_length = getattr(cfg, "max_completion_length", 3072)
    temperature = getattr(cfg, "temperature", 0.6)
    beta = getattr(cfg, "beta", 0.005)
    top_p = getattr(cfg, "top_p", 0.95)

    # vLLM knobs
    use_vllm = getattr(cfg, "use_vllm", False)
    vllm_mode = getattr(cfg, "vllm_mode", "colocate")

    # Wandb configuration
    report_to = ["wandb"] if getattr(cfg, "report_to") is None else getattr(cfg, "report_to")
    run_name = getattr(cfg, "run_name", None)
    
    # Auto-generate run name if not provided
    if run_name is None:
        model_name = os.path.basename(cfg.model_name)
        target_task = cfg.target_task
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"grpo_{model_name}_{target_task}_lr{learning_rate}_bs{per_device_train_batch_size}_gen{num_generations}_seed{training_seed}_{timestamp}"
    
    if output_dir_override:
        # Add output dir suffix to make runs identifiable
        run_name = f"{run_name}_{os.path.basename(output_dir_override)}"

    args = GRPOConfig(
        # --- core TrainingArguments fields ---
        output_dir=output_dir_override or cfg.output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        logging_steps=cfg.log_every,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=getattr(cfg, "save_strategy", "no"),
        save_steps=getattr(cfg, "save_steps", None) if getattr(cfg, "save_strategy", "no") == "steps" else None,
        save_total_limit=getattr(cfg, "save_total_limit", None),
        bf16=bf16,
        fp16=fp16,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=cfg.num_workers,
        dataloader_pin_memory=cfg.pin_memory,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=run_name,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=cfg.drop_last,
        seed=training_seed,
        max_grad_norm=max_grad_norm,

        # --- GRPO / generation ---
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=temperature,
        beta=beta,
        top_p=top_p,

        # --- gradient checkpointing ---
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,

        # --- vLLM integration ---
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_server_base_url=getattr(cfg, "vllm_server_base_url", None),
        vllm_importance_sampling_correction=getattr(cfg, "vllm_importance_sampling_correction", False),
    )

    # Note: GRPOMultiTaskTrainer will override args.max_steps based on global token budget.
    trainer = GRPOMultiTaskTrainer(
        cfg=local_cfg,
        args=args,
        data_module=data_module,
        tasks_override=tasks_override,
        mix_override=mix_override,
    )
    
    # Save config to wandb if reporting to wandb
    if "wandb" in report_to and trainer.args.process_index == 0:
        import wandb
        if wandb.run is not None:
            # Convert cfg to dict for wandb
            cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
            wandb.config.update(cfg_dict, allow_val_change=True)
    
    return trainer


# ----------------------------------------------------------------------
# SFT MULTITASK TRAINER (INSTRUCTION FINETUNING)
# ----------------------------------------------------------------------

def make_sft_trainer(
    *,
    cfg: Any,
    tasks_override: Optional[List[str]] = None,
    mix_override: Optional[Dict[str, float]] = None,
    output_dir_override: Optional[str] = None,
    data_module: Optional[Any] = None,
    seed_offset: int = 0,
) -> SFTMultiTaskTrainer:
    """
    Build an SFTMultiTaskTrainer for multitask instruction finetuning.

    Important knobs routed from cfg:
      - standard training args (lr, weight_decay, warmup_ratio, etc.)
      - SFT-specific: max_seq_length, packing
      - stability: gradient_checkpointing, gradient_accumulation_steps,
        bf16/fp16, max_grad_norm
    """

    # ---- per-trainer seed ----
    base_seed = getattr(cfg, "seed", 42)
    training_seed = base_seed + seed_offset
    set_seed(training_seed)

    # ---- eval cadence ----
    if int(cfg.eval_every) == -1:
        eval_strategy = "no"
        eval_steps = None
    else:
        eval_strategy = "steps"
        eval_steps = cfg.eval_every

    # ---- shallow copy cfg if we override tasks ----
    local_cfg = cfg if tasks_override is None else copy.deepcopy(cfg)
    if tasks_override is not None:
        local_cfg.tasks = list(tasks_override)

    # ---- SFTConfig construction ----
    # Standard training knobs
    learning_rate = getattr(cfg, "lr", 1e-5)
    weight_decay = getattr(cfg, "weight_decay", 0.0)
    warmup_ratio = getattr(cfg, "warmup_ratio", 0.0)
    lr_scheduler_type = getattr(cfg, "lr_scheduler_type", "constant")
    optim = getattr(cfg, "optim", "adamw_torch")
    max_steps = getattr(cfg, "max_steps", 1000)
    num_train_epochs = getattr(cfg, "num_train_epochs", 1)

    # Batch / precision / stability
    per_device_train_batch_size = getattr(cfg, "per_device_train_batch_size", cfg.train_batch_size)
    per_device_eval_batch_size = getattr(cfg, "per_device_eval_batch_size", 
                                          getattr(cfg, "eval_batch_size", per_device_train_batch_size))
    grad_accum = getattr(cfg, "grad_accum_steps", 1)
    bf16 = getattr(cfg, "bf16", False)
    fp16 = getattr(cfg, "fp16", False)
    gradient_checkpointing = getattr(cfg, "gradient_checkpointing", False)
    gradient_checkpointing_kwargs = getattr(
        cfg,
        "gradient_checkpointing_kwargs",
        {"use_reentrant": False},
    )
    max_grad_norm = getattr(cfg, "max_grad_norm", 1.0)

    # SFT-specific knobs
    max_seq_length = getattr(cfg, "max_seq_length", 2048)
    packing = getattr(cfg, "packing", False)
    
    # Wandb configuration
    report_to = ["wandb"] if getattr(cfg, "report_to", None) is None else getattr(cfg, "report_to")
    run_name = getattr(cfg, "run_name", None)
    
    # Auto-generate run name if not provided
    if run_name is None:
        model_name = os.path.basename(cfg.model_name)
        target_task = getattr(cfg, "target_task", "sft")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"sft_{model_name}_{target_task}_lr{learning_rate}_bs{per_device_train_batch_size}_seed{training_seed}_{timestamp}"
    
    if output_dir_override:
        # Add output dir suffix to make runs identifiable
        run_name = f"{run_name}_{os.path.basename(output_dir_override)}"

    args = SFTConfig(
        # --- core TrainingArguments fields ---
        output_dir=output_dir_override or cfg.output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        logging_steps=cfg.log_every,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=getattr(cfg, "save_strategy", "no"),
        save_steps=getattr(cfg, "save_steps", None) if getattr(cfg, "save_strategy", "no") == "steps" else None,
        save_total_limit=getattr(cfg, "save_total_limit", None),
        bf16=bf16,
        fp16=fp16,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=cfg.num_workers,
        dataloader_pin_memory=cfg.pin_memory,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=run_name,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=cfg.drop_last,
        seed=training_seed,
        max_grad_norm=max_grad_norm,

        # --- gradient checkpointing ---
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,

        # --- SFT-specific ---
        max_length=max_seq_length,
        packing=packing,
    )

    # Note: SFTMultiTaskTrainer will override args.max_steps based on global token budget.
    trainer = SFTMultiTaskTrainer(
        cfg=local_cfg,
        args=args,
        data_module=data_module,
        tasks_override=tasks_override,
        mix_override=mix_override,
    )
    
    # Save config to wandb if reporting to wandb
    if "wandb" in report_to and trainer.args.process_index == 0:
        import wandb
        if wandb.run is not None:
            # Convert cfg to dict for wandb
            cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
            wandb.config.update(cfg_dict, allow_val_change=True)
    
    return trainer