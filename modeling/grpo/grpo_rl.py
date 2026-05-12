from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.generation.utils import GenerationMixin

from transformers import BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType as PeftTaskType


@dataclass
class GRPORLModelConfig:
    """
    Lightweight config for RL (GRPO) causal LM models.

    This is intentionally *not* a HuggingFace PretrainedConfig; it's just a
    convenient container for our experiment YAML / DotDict settings.

    Key flags:
        - model_name: base HF checkpoint
        - load_in_4bit / load_in_8bit: bitsandbytes quantization
        - torch_dtype: "bfloat16", "float16", "float32" or a torch.dtype
        - device_map: "auto" or a mapping (passed to from_pretrained)
        - use_lora: whether to wrap with LoRA
        - lora_r, lora_alpha, lora_dropout, lora_target_modules, lora_task_type:
              LoRA hyperparameters (we KEEP your default modules + dropout + task_type)
        - gradient_checkpointing: enable gradient checkpointing on the base model
    """

    # Base model
    model_name: str

    # Quantization / dtype
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: Optional[Union[str, torch.dtype]] = "bfloat16"
    device_map: Optional[Union[str, Dict[str, int]]] = None  # None = let Trainer handle device placement

    # LoRA
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = field(default_factory=lambda: [
        "embed_tokens",
        "q_proj",
        "v_proj",
        "o_proj",
        "k_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ])
    lora_task_type: str = "CAUSAL_LM"  # mapped to PeftTaskType.CAUSAL_LM

    # Training-time tweaks
    gradient_checkpointing: bool = False

    # Extra kwargs to pass into AutoModelForCausalLM.from_pretrained
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


def _resolve_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    s = str(dtype).lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    # fallback
    return None


class GRPORLModel(PreTrainedModel, GenerationMixin):
    """
    Thin wrapper around a causal LM (AutoModelForCausalLM) for GRPO-style RL.

    Responsibilities:
      - Load the base model with optional 4/8-bit quantization.
      - Optionally wrap with LoRA using your default target modules & task_type.
      - Expose `self.model` as the object passed into GRPOTrainer.

    Tokenizer is intentionally handled outside (in the trainer mixin), so this
    class focuses purely on model weights & trainability.
    """

    def __init__(self, cfg: GRPORLModelConfig):
        base_config = AutoConfig.from_pretrained(cfg.model_name)
        super().__init__(base_config)
        self.cfg = cfg
        self.model = self._build_model()

    # ------------------------------------------------------------------
    # Core builder
    # ------------------------------------------------------------------
    def _build_model(self):
        cfg = self.cfg

        # ---- Dtype / quantization setup ----
        torch_dtype = _resolve_dtype(cfg.torch_dtype)
        quantization_config = None
        model_kwargs: Dict[str, Any] = dict(cfg.model_kwargs)  # shallow copy

        if (cfg.load_in_4bit or cfg.load_in_8bit) and BitsAndBytesConfig is not None:
            # bitsandbytes quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=cfg.load_in_4bit,
                load_in_8bit=cfg.load_in_8bit,
                bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config
            if cfg.device_map is not None:
                model_kwargs["device_map"] = cfg.device_map
        else:
            # full-precision / mixed precision
            if torch_dtype is not None:
                model_kwargs["dtype"] = torch_dtype
            if cfg.device_map is not None:
                model_kwargs["device_map"] = cfg.device_map

        # ---- Load base causal LM ----
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

        # ---- Gradient checkpointing ----
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        # ---- LoRA (optional) ----
        if cfg.use_lora:
            if LoraConfig is None or get_peft_model is None:
                raise ImportError(
                    "peft is required for LoRA but is not installed. "
                    "Install with `pip install peft` or disable use_lora."
                )

            # For k-bit training, we should prepare the model first
            if (cfg.load_in_4bit or cfg.load_in_8bit) and prepare_model_for_kbit_training is not None:
                model = prepare_model_for_kbit_training(model)

            target_modules = cfg.lora_target_modules or [
                "q_proj",
                "v_proj",
                "o_proj",
                "k_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ]

            # Map string task type to PeftTaskType enum if available
            task_type_enum = None
            if PeftTaskType is not None:
                try:
                    task_type_enum = getattr(PeftTaskType, cfg.lora_task_type.upper())
                except AttributeError:
                    task_type_enum = PeftTaskType.CAUSAL_LM
            else:
                # Fallback: peft not fully imported, but we already checked above
                pass

            lora_cfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=target_modules,
                task_type=task_type_enum,
                bias="none",
            )

            model = get_peft_model(model, lora_cfg)
            model._mark_only_adapters_as_trainable(model=model) # let deepspeed/Trainer see only LoRA params
            model.print_trainable_parameters()

        return model

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def is_lora(self) -> bool:
        """Heuristic to check if the model has LoRA adapters."""
        return hasattr(self.model, "peft_config")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        # TRL/Accelerate call this on the (un)wrapped model
        return self.model.generate(*args, **kwargs)

    # HF GenerationMixin may call this on the model; delegate if present
    def prepare_inputs_for_generation(self, *args, **kwargs):
        if hasattr(self.model, "prepare_inputs_for_generation"):
            return self.model.prepare_inputs_for_generation(*args, **kwargs)
        return super().prepare_inputs_for_generation(*args, **kwargs)

    # --- gradient checkpointing: override and delegate ---

    def _gc_enable_inner(self, model, kwargs=None):
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                # Newer HF supports the kwarg
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
            except TypeError:
                model.gradient_checkpointing_enable()
        # HF convention: disable KV cache when using GC
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    def _gc_disable_inner(self, model):
        if hasattr(model, "gradient_checkpointing_disable"):
            try:
                model.gradient_checkpointing_disable()
            except TypeError:
                pass
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            # let callers re-enable if they want; no-op is fine too
            model.config.use_cache = True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # enable on the wrapped model
        self._gc_enable_inner(self.model, gradient_checkpointing_kwargs)
        # if PEFT is used, also enable on the base model
        if PeftModel is not None and isinstance(self.model, PeftModel):
            base = self.model.get_base_model()
            self._gc_enable_inner(base, gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self._gc_disable_inner(self.model)
        if PeftModel is not None and isinstance(self.model, PeftModel):
            base = self.model.get_base_model()
            self._gc_disable_inner(base)

    # Prefer default state_dict for checkpointing (don’t rename saved keys)
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)