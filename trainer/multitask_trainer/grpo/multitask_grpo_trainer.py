from __future__ import annotations
from typing import Dict, List, Optional, Any

import os
import json
import shlex
import time
import subprocess
from pathlib import Path
from trl import GRPOTrainer, GRPOConfig
import torch.distributed as dist

from trainer.multitask_trainer.base import MultiTaskTrainerBase
from data.base_module import TaskSpec

from data.base_module import DataModuleConfig
from data.big_math_gsm8k.module import BigMathRLGSM8KGoogleTransRLDataModule
from data.big_math_gsm8k.formatter import BIG_MATH_RL_GSM8K_GOOGLETRANS_RL_FORMATTER
from trainer.multitask_trainer.grpo.big_math_gsm8k.collator import BigMathRLGSM8KGoogleTransRLGRPOCollator
from trainer.multitask_trainer.grpo.big_math_gsm8k.rl_rewards import get_big_math_gsm8k_reward_funcs


from modeling.grpo.grpo_rl import GRPORLModel, GRPORLModelConfig
from data.multitask_iterable import MultiTaskOnTheFlyDataset

from transformers import TrainerCallback

class LoRADeltaCallback(TrainerCallback):
    """
    Debugger.
    Tracks ONE LoRA-A matrix and ONE LoRA-B matrix:
      - logs max|Δ| / mean|Δ| right after optimizer.step()
      - logs parameter magnitude stats (L2, mean|x|, max|x|)
    """
    def __init__(
        self,
        every: int = 1,
        pick_a_substr: str = "lora_A.default.weight",
        pick_b_substr: str = "lora_B.default.weight",
        also_print_lr: bool = True,
    ):
        self.every = every
        self.pick_a_substr = pick_a_substr
        self.pick_b_substr = pick_b_substr
        self.also_print_lr = also_print_lr

        self._a_name = None
        self._b_name = None
        self._prev_a = None
        self._prev_b = None

    def _inner(self, model):
        m = model
        if hasattr(m, "module"):  # DDP
            m = m.module
        if hasattr(m, "model"):   # your GRPORLModel wrapper
            m = m.model
        return m

    def _pick_param_by_substr(self, model, substr: str):
        m = self._inner(model)
        for n, p in m.named_parameters():
            if substr in n and p.requires_grad:
                return n, p
        return None, None

    def _pick_any_lora(self, model, which: str):
        assert which in ("A", "B")
        m = self._inner(model)
        key = "lora_A" if which == "A" else "lora_B"
        for n, p in m.named_parameters():
            if key in n and p.requires_grad:
                return n, p
        return None, None

    @staticmethod
    def _mag_stats(t_cpu_fp32):
        # t_cpu_fp32: torch.Tensor on CPU float32
        abs_t = t_cpu_fp32.abs()
        return {
            "l2": t_cpu_fp32.norm().item(),
            "mean_abs": abs_t.mean().item(),
            "max_abs": abs_t.max().item(),
        }

    def _get_lr(self, kwargs) -> float:
        if not self.also_print_lr:
            return float("nan")
        opt = kwargs.get("optimizer", None)
        if opt is None:
            return float("nan")
        try:
            return float(opt.param_groups[0]["lr"])
        except Exception:
            return float("nan")

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]

        # pick A
        a_name, a_p = self._pick_param_by_substr(model, self.pick_a_substr)
        if a_p is None:
            a_name, a_p = self._pick_any_lora(model, "A")

        # pick B
        b_name, b_p = self._pick_param_by_substr(model, self.pick_b_substr)
        if b_p is None:
            b_name, b_p = self._pick_any_lora(model, "B")

        self._a_name, self._b_name = a_name, b_name

        if a_p is None and b_p is None:
            print("[LoRAΔ] Could not find any trainable LoRA-A or LoRA-B param.")
            return

        if a_p is not None:
            a0 = a_p.detach().float().cpu().clone()
            self._prev_a = a0
            s = self._mag_stats(a0)
            print(
                f"[LoRAΔ] Tracking A: {self._a_name} | "
                f"l2={s['l2']:.8e} mean|x|={s['mean_abs']:.8e} max|x|={s['max_abs']:.8e}"
            )
        else:
            print("[LoRAΔ] WARNING: No trainable LoRA-A found.")

        if b_p is not None:
            b0 = b_p.detach().float().cpu().clone()
            self._prev_b = b0
            s = self._mag_stats(b0)
            print(
                f"[LoRAΔ] Tracking B: {self._b_name} | "
                f"l2={s['l2']:.8e} mean|x|={s['mean_abs']:.8e} max|x|={s['max_abs']:.8e}"
            )
        else:
            print("[LoRAΔ] WARNING: No trainable LoRA-B found.")

    def on_optimizer_step(self, args, state, control, **kwargs):
        # called right after optimizer.step() in HF Trainer
        if state.global_step % self.every != 0:
            return

        model = kwargs["model"]
        m = self._inner(model)

        named = dict(m.named_parameters())

        lr = self._get_lr(kwargs)
        lr_str = f" lr={lr:.3e}" if self.also_print_lr and lr == lr else ""  # lr==lr filters NaN

        # --- A ---
        if self._a_name is not None and self._prev_a is not None:
            p = named.get(self._a_name, None)
            if p is None:
                print(f"[LoRAΔ] Missing tracked A param: {self._a_name}")
            else:
                cur = p.detach().float().cpu()
                d = (cur - self._prev_a)
                max_abs = d.abs().max().item()
                mean_abs = d.abs().mean().item()
                s = self._mag_stats(cur)
                print(
                    f"[LoRAΔ opt_step={state.global_step}]{lr_str} A "
                    f"max|Δ|={max_abs:.8e} mean|Δ|={mean_abs:.8e} | "
                    f"l2={s['l2']:.8e} mean|x|={s['mean_abs']:.8e} max|x|={s['max_abs']:.8e} "
                    f"({self._a_name})"
                )
                self._prev_a = cur.clone()

        # --- B ---
        if self._b_name is not None and self._prev_b is not None:
            p = named.get(self._b_name, None)
            if p is None:
                print(f"[LoRAΔ] Missing tracked B param: {self._b_name}")
            else:
                cur = p.detach().float().cpu()
                d = (cur - self._prev_b)
                max_abs = d.abs().max().item()
                mean_abs = d.abs().mean().item()
                s = self._mag_stats(cur)
                print(
                    f"[LoRAΔ opt_step={state.global_step}]{lr_str} B "
                    f"max|Δ|={max_abs:.8e} mean|Δ|={mean_abs:.8e} | "
                    f"l2={s['l2']:.8e} mean|x|={s['mean_abs']:.8e} max|x|={s['max_abs']:.8e} "
                    f"({self._b_name})"
                )
                self._prev_b = cur.clone()



class GRPOMultiTaskTrainer(MultiTaskTrainerBase, GRPOTrainer):
    """
    Multitask GRPO trainer for MATH RL, mirroring HFMultiTaskTrainer's behavior:

      - cfg is the same DotDict YAML config
      - optional data_module reuse with cached loaders
      - tasks_override / mix_override behave like in MLM
      - token-budget -> max_steps uses the same shared helper
      - train pipeline uses a map-style MultiTaskOnTheFlyDataset so that
        TRL's RepeatedSampler / GRPO machinery works correctly.
    """

    def __init__(
        self,
        *,
        cfg,  # DotDict YAML config
        tasks_override: Optional[List[str]] = None,
        mix_override: Optional[Dict[str, float]] = None,
        data_module: Optional[Any] = None,
        **kwargs,
    ):

        # ---------------- tokenizer ----------------
        tok = self._init_tokenizer(cfg)

        # ---------------- data module ----------------
        # First try to reuse an existing one (and snapshot original loaders)
        dm = self._init_or_reuse_data_module(data_module)

        if dm is None:
            # Determine task type from config
            task_type = getattr(cfg, "task_type", "math-rl")
            
            # Select appropriate components based on task_type
            if task_type == "big-math-gsm8k":
                data_module_cls = BigMathRLGSM8KGoogleTransRLDataModule
                formatter = BIG_MATH_RL_GSM8K_GOOGLETRANS_RL_FORMATTER
                collator = BigMathRLGSM8KGoogleTransRLGRPOCollator(tokenizer=tok)
            else:
                raise ValueError(
                    f"Unknown task_type '{task_type}'. "
                    f"Supported: 'big-math-gsm8k'"
                )

            # Build task specs with few-shot configuration
            specs: List[TaskSpec] = []
            few_overrides = getattr(cfg, "few_shot", None) or {}
            k_shot_global = getattr(cfg, "k_shot", None)

            for t_name in cfg.tasks:
                if t_name in few_overrides:
                    k = few_overrides[t_name]
                elif k_shot_global is not None and t_name != cfg.target_task:
                    k = k_shot_global
                else:
                    k = None
                specs.append(
                    TaskSpec(
                        name=t_name,
                        k_shot=k,
                        k_shot_seed=getattr(cfg, "few_shot_seed", 42),
                    )
                )

            dm_cfg = DataModuleConfig(
                datadir=getattr(cfg, "datadir", None),
                save_kshot_locally=getattr(cfg, "save_kshot_locally", True),
                num_workers=cfg.num_workers,
                # For GRPO, use train_batch_size=**unique prompts** per batch
                train_batch_size=cfg.train_batch_size,
                eval_batch_size=getattr(cfg, "per_device_eval_batch_size", None) or cfg.eval_batch_size,
                pin_memory=cfg.pin_memory,
                drop_last=cfg.drop_last,
            )

            # Instantiate the appropriate data module
            dm_kwargs = {
                "tasks": specs,
                "cfg": dm_cfg,
                "formatter": formatter,
                "collator": collator,
                "tokenizer": tok,
            }
            
            dm = data_module_cls(**dm_kwargs)
            
            dm.prepare()
            dm.setup_dataloaders()

        # ---------------- task selection + loader filtering ----------------
        task_names, train_loaders, eval_loaders = self._select_tasks_and_filter_loaders(
            dm=dm,
            tasks_override=tasks_override,
        )
        self.task_names = task_names

        # ---------------- mixture over tasks ----------------
        mix = self._compute_mix(cfg, task_names, mix_override, train_loaders=train_loaders)
        self.mix = mix

        # ---------------- planned max_steps ----------------
        planned_max_steps, per_task_steps = self._plan_max_steps(cfg, task_names, mix)

        if "args" not in kwargs:
            raise ValueError("GRPOMultiTaskTrainer expects `args=GRPOConfig(...)` in kwargs.")
        args: GRPOConfig = kwargs["args"]

        # Let the token-budget logic control the actual step count
        object.__setattr__(args, "max_steps", planned_max_steps)
        # We pass custom fields (prompt, answer, etc.), so don't drop them
        object.__setattr__(args, "remove_unused_columns", False)

        # ---------------- multitask on-the-fly dataset (map-style) ----------------
        train_dataset = self._build_grpo_multitask_dataset(
            train_loaders=train_loaders,
            mix=mix,
            planned_max_steps=planned_max_steps,
            cfg=cfg,
            tok=tok,
        )

        # Keep for potential debugging / logging
        self._train_dataset = train_dataset
        self._train_loaders = train_loaders
        self._eval_loaders = eval_loaders
        self._per_task_steps = per_task_steps

        # ---------------- target eval dataset ----------------
        target_eval_ds = None
        if cfg.target_task in eval_loaders:
            target_eval_ds = eval_loaders[cfg.target_task].dataset

        # ---------------- model construction ----------------
        lora_target_modules_default = [
            "embed_tokens",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]

        model_config = GRPORLModelConfig(
            model_name=cfg.model_name,
            # Quantization / dtype
            load_in_4bit=getattr(cfg, "load_in_4bit", False),
            load_in_8bit=getattr(cfg, "load_in_8bit", False),
            torch_dtype=getattr(cfg, "torch_dtype", "bfloat16"),
            device_map=getattr(cfg, "device_map", None),
            # LoRA
            use_lora=getattr(cfg, "use_lora", True),
            lora_r=getattr(cfg, "lora_r", 64),
            lora_alpha=getattr(cfg, "lora_alpha", 128),
            lora_dropout=getattr(cfg, "lora_dropout", 0.05),
            lora_target_modules=getattr(cfg, "lora_target_modules", lora_target_modules_default),
            lora_task_type=getattr(cfg, "lora_task_type", "CAUSAL_LM"),
            # Training
            gradient_checkpointing=getattr(cfg, "gradient_checkpointing", False),
        )
        hf_model = GRPORLModel(model_config)

        # ---------------- reward function(s) ----------------
        reward_funcs = self._build_reward_functions(cfg)

        # ---------------- store state for later use ----------------
        self.cfg = cfg
        self.tok = tok
        self.dm = dm

        print(f"[GRPO-MT] Tasks: {self.task_names}")
        print(f"[GRPO-MT] Mix: {self.mix}")
        print(f"[GRPO-MT] Planned max_steps: {planned_max_steps}")

        # ---------------- Patch vLLM compatibility ----------------
        if args.use_vllm:
            self._patch_vllm_compatibility()
            self._launch_vllm_server(cfg)      # no dist.barrier inside
            self._patch_trl_vllm_client_rank0_only()

        # ---------------- call base GRPOTrainer ----------------
        # NOTE: we pass train_dataset (map-style). We DO NOT override
        # get_train_dataloader, so TRL's RepeatedSampler + GRPO batching
        # remain intact.
        super().__init__(
            model=hf_model,
            processing_class=tok,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=target_eval_ds,
            **{k: v for k, v in kwargs.items() if k != "args"},
        )
        
        # Set the data_collator after initialization
        # IMPORTANT: Use GRPOCollator that converts chat-format prompts to strings
        self.data_collator = dm.collator

    # ---------------- dataset builder (GRPO-specific) ---------------- #
    def _build_grpo_multitask_dataset(
        self,
        train_loaders: Dict,
        mix: Dict[str, float],
        planned_max_steps: int,
        cfg,
        tok,
    ):
        """
        Build multitask on-the-fly dataset for GRPO training.
        
        GRPO uses per_device_train_batch_size, which needs to be multiplied
        by world_size and grad_accum_steps to get global batch size.
        """
        mt_seed = int(getattr(cfg, "seed", 42))

        # Compute global batch size (matching base.py logic)
        if getattr(cfg, "per_device_train_batch_size", None) is not None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            global_bs = (
                cfg.per_device_train_batch_size
                * world_size
                * getattr(cfg, "grad_accum_steps", 1)
            )
        else:
            # train_batch_size is already global, don't multiply it
            global_bs = getattr(cfg, "train_batch_size", 1)

        virtual_length = planned_max_steps * global_bs

        return MultiTaskOnTheFlyDataset(
            train_loaders=train_loaders,
            mix=mix,
            length=virtual_length,
            seed=mt_seed,
            tokenizer=tok,
        )
    
    # ---------------- reward helper (RL-specific) ---------------- #
    @staticmethod
    def _build_reward_functions(cfg):
        """
        Build reward functions based on task_type and reward_name from config.
        
        For math-rl:
          - math_boxed_accuracy (default)
        
        For geo-factual:
          - geo_factual_combined (default): format + correctness_fuzzy + language_matching
          - geo_factual_correctness: correctness_fuzzy only
          - geo_factual_format: format only
          - geo_factual_language: language_matching only
        """
        task_type = getattr(cfg, "task_type", "math-rl")
        
        if task_type == "big-math-gsm8k":
            return get_big_math_gsm8k_reward_funcs()
        else:
            raise ValueError(
                f"Unknown task_type '{task_type}'. "
                f"Supported: 'math-rl', 'geo-factual', 'multilingual-math', 'limr'"
            )

    def _patch_trl_vllm_client_rank0_only(self):
        """
        Ensure only training rank0 initializes the vLLM NCCL communicator and updates weights.
        Other ranks can still use HTTP generate/chat but will NOT touch NCCL group.
        """
        try:
            from trl.extras.vllm_client import VLLMClient

            rank = int(os.environ.get("RANK", "0"))
            is_rank0 = (rank == 0)

            # --- Patch init_communicator ---
            _orig_init_comm = VLLMClient.init_communicator

            def _patched_init_communicator(self, device=0):
                if not is_rank0:
                    # Skip communicator init on non-rank0
                    return None

                # IMPORTANT: pass the *current* cuda device index (local) to avoid device mismatch
                if isinstance(device, int):
                    # if device=0, it means "current visible cuda:0"
                    # safer: always use torch.cuda.current_device()
                    import torch
                    device = torch.cuda.current_device()

                return _orig_init_comm(self, device=device)

            VLLMClient.init_communicator = _patched_init_communicator

            # --- Patch weight update calls ---
            _orig_update_named = VLLMClient.update_named_param
            _orig_update_all = VLLMClient.update_model_params

            def _patched_update_named_param(self, name, weights):
                if not is_rank0:
                    return None
                return _orig_update_named(self, name, weights)

            def _patched_update_model_params(self, model):
                if not is_rank0:
                    return None
                return _orig_update_all(self, model)

            VLLMClient.update_named_param = _patched_update_named_param
            VLLMClient.update_model_params = _patched_update_model_params

            print("[vLLM/TRL] Patched VLLMClient: communicator + weight updates are rank0-only.")
        except Exception as e:
            print(f"[vLLM/TRL] WARNING: could not patch VLLMClient rank0-only ({e})")

    @staticmethod
    def _patch_vllm_compatibility():
        """
        Patch vLLM EngineArgs to handle compatibility issues:
        1. Filter unsupported parameters like 'model_impl'
        2. Fix tensor_parallel_size for DDP multi-GPU training
        """
        try:
            from vllm import EngineArgs
            import inspect

            original_init = EngineArgs.__init__
            sig = inspect.signature(original_init)
            supported_params = set(sig.parameters.keys())

            def patched_init(self, **kwargs):
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
                unsupported = set(kwargs.keys()) - supported_params
                if unsupported:
                    print(f"[vLLM compat] Filtering out unsupported parameters: {unsupported}")

                world_size = int(os.environ.get("WORLD_SIZE", "1"))
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))

                if world_size > 1:
                    # In DDP, you likely want 1 TP per rank (or leave TP config to server)
                    if "tensor_parallel_size" in filtered_kwargs:
                        original_tp = filtered_kwargs["tensor_parallel_size"]
                        if original_tp != 1:
                            print(
                                f"[vLLM compat] DDP detected (world_size={world_size}). "
                                f"Overriding tensor_parallel_size from {original_tp} to 1"
                            )
                        filtered_kwargs["tensor_parallel_size"] = 1

                    print(f"[vLLM compat] Rank {local_rank}/{world_size} initializing vLLM with TP=1")

                return original_init(self, **filtered_kwargs)

            EngineArgs.__init__ = patched_init
            print("[vLLM compat] Successfully patched EngineArgs")

        except Exception as e:
            print(f"[vLLM compat] Warning: Could not patch EngineArgs: {e}")
            print("[vLLM compat] Proceeding without patch - may encounter errors if version mismatch exists")
    
    def _launch_vllm_server(self, cfg):
        """
        Launch vLLM server via screen (only on training rank 0),
        and sync other ranks using ONLY filesystem + port polling (no dist.barrier).

        This avoids NCCL/device-mapping hangs when the vLLM GPU is not part of DDP.
        """
        import socket
        import time

        rank = int(os.environ.get("RANK", "0"))

        port = int(getattr(cfg, "vllm_port", 10245))
        gpu = str(getattr(cfg, "vllm_gpu", "0"))  # physical GPU index for vLLM (you want "0")
        model = getattr(cfg, "model_name", None) or cfg.model_name

        script_path = (
            Path(__file__).parent.parent.parent.parent / "scripts" / "grpo" / "vllm_server.sh"
        )

        ready_file = Path(f"/tmp/vllm_server_{port}.ready")
        fail_file = Path(f"/tmp/vllm_server_{port}.fail")

        STARTUP_TIMEOUT_S = int(getattr(cfg, "vllm_startup_timeout", 300))
        POLL_INTERVAL_S = 0.5

        def _port_open() -> bool:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            try:
                return s.connect_ex(("127.0.0.1", port)) == 0
            finally:
                s.close()

        def _wait_for_ready():
            t0 = time.time()
            last_log = 0.0
            while True:
                if fail_file.exists():
                    msg = fail_file.read_text(errors="ignore")
                    raise RuntimeError(f"[vLLM-launch rank {rank}] vLLM failed: {msg}")

                if ready_file.exists() and _port_open():
                    return

                elapsed = time.time() - t0
                if elapsed > STARTUP_TIMEOUT_S:
                    raise TimeoutError(
                        f"[vLLM-launch rank {rank}] Timed out waiting for vLLM on port {port} "
                        f"after {STARTUP_TIMEOUT_S}s"
                    )

                if elapsed - last_log > 15:
                    last_log = elapsed
                    print(f"[vLLM-launch rank {rank}] Waiting for vLLM ready... ({int(elapsed)}s)")
                time.sleep(POLL_INTERVAL_S)

        # ---------------- rank 0 launches ----------------
        if rank == 0:
            if not script_path.exists():
                msg = f"Script not found at {script_path}"
                print(f"[vLLM-launch rank {rank}] ERROR: {msg}")
                fail_file.write_text(msg)
                return

            # Clear stale markers
            for p in (ready_file, fail_file):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

            # Kill existing screen session if any
            check_session = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
            if "vllm-server" in (check_session.stdout or ""):
                print(f"[vLLM-launch rank {rank}] Killing existing vllm-server screen session")
                subprocess.run(["screen", "-S", "vllm-server", "-X", "quit"])
                time.sleep(2)

            print(f"[vLLM-launch rank {rank}] Launching vLLM server in screen session 'vllm-server'")
            print(f"[vLLM-launch rank {rank}] PORT={port}, GPU={gpu}, MODEL={model}")

            # IMPORTANT: do NOT wipe env completely; keep HOME + HF caches.
            path = os.environ.get("PATH", "")
            ld_lib = os.environ.get("LD_LIBRARY_PATH", "")
            home = os.environ.get("HOME", "")
            hf_home = os.environ.get("HF_HOME", "")
            tr_cache = os.environ.get("TRANSFORMERS_CACHE", "")

            # Build clean environment without distributed training variables
            # This prevents vLLM from inheriting WORLD_SIZE, RANK, etc.
            inner_cmd = (
                "/usr/bin/env -i "
                f"PATH={shlex.quote(path)} "
                f"LD_LIBRARY_PATH={shlex.quote(ld_lib)} "
                f"HOME={shlex.quote(home)} "
                f"HF_HOME={shlex.quote(hf_home)} "
                f"TRANSFORMERS_CACHE={shlex.quote(tr_cache)} "
                f"CUDA_VISIBLE_DEVICES={gpu} "
                # --- add these ---
                f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'eno1')} "
                "NCCL_IB_DISABLE=1 "
                "NCCL_SHM_DISABLE=1 "
                "NCCL_P2P_DISABLE=1 "
                # optional but helpful:
                "NCCL_DEBUG=INFO "
                "NCCL_ASYNC_ERROR_HANDLING=1 "
                # ------------------
                f"GPU={gpu} "
                f"PORT={port} "
                f"MODEL={shlex.quote(model)} "
                f"bash {shlex.quote(str(script_path))}"
            )

            proc = subprocess.run(
                ["screen", "-dmS", "vllm-server", "bash", "-c", inner_cmd],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                err = proc.stderr or "unknown error"
                print(f"[vLLM-launch rank {rank}] Failed to launch screen: {err}")
                fail_file.write_text(err)
                return

            # Wait for port then write ready
            t0 = time.time()
            last_log = 0.0
            while True:
                if _port_open():
                    print(f"[vLLM-launch rank {rank}] Port {port} is active after {time.time()-t0:.1f}s")
                    ready_file.write_text("ok")
                    break
                elapsed = time.time() - t0
                if elapsed > STARTUP_TIMEOUT_S:
                    msg = f"Port {port} not active after {STARTUP_TIMEOUT_S}s"
                    print(f"[vLLM-launch rank {rank}] ERROR: {msg}")
                    fail_file.write_text(msg)
                    return
                if elapsed - last_log > 15:
                    last_log = elapsed
                    print(f"[vLLM-launch rank {rank}] Still waiting for port {port}... ({int(elapsed)}s)")
                time.sleep(POLL_INTERVAL_S)

        else:
            print(f"[vLLM-launch rank {rank}] Waiting for rank 0 to start vLLM...")

        # Everyone waits (no NCCL barrier)
        _wait_for_ready()
        print(f"[vLLM-launch rank {rank}] vLLM server is up and reachable on port {port}")

    def _cleanup_vllm_server(self):
        """Kill vLLM server screen session when training is done."""
        rank = int(os.environ.get("RANK", "0"))
        
        if rank == 0:
            port = str(getattr(self.cfg, "vllm_port", 10245))
            lock_file = Path(f"/tmp/vllm_server_{port}.lock")
            
            # Check if screen session exists
            check_session = subprocess.run(
                ["screen", "-ls"],
                capture_output=True,
                text=True,
            )
            if "vllm-server" in check_session.stdout:
                print(f"[vLLM-cleanup rank {rank}] Killing vllm-server screen session")
                subprocess.run(["screen", "-S", "vllm-server", "-X", "quit"])
                time.sleep(1)
            
            # Remove lock file
            if lock_file.exists():
                print(f"[vLLM-cleanup rank {rank}] Removing lock file")
                lock_file.unlink()
            
            print(f"[vLLM-cleanup rank {rank}] Cleanup complete")
    
    def save_model(self, output_dir: Optional[str] = None):

        if not self.accelerator.is_main_process:
            return

        if output_dir is None:
            output_dir = self.args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1) unwrap accelerator/DDP
        m = self.model
        if hasattr(m, "module"):   # DDP
            m = m.module

        # 2) unwrap your wrapper
        inner = m.model if hasattr(m, "model") else m

        # 3) if PEFT merged, unmerge before saving adapters
        if hasattr(inner, "is_merged") and inner.is_merged:
            inner.unmerge_adapter()

        inner.save_pretrained(output_dir, safe_serialization=False)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        ap = os.path.join(output_dir, "adapter_model.bin")
        if os.path.exists(ap):
            print(f"[save_model] wrote adapter_model.bin size={os.path.getsize(ap)}")
        else:
            print(f"[save_model] WARNING: adapter_model.bin not found in {output_dir}")
        
        # 4) Save run_info.json in the parent directory (for lm-eval to read model_name)
        # This is saved once at the output_dir level (parent of checkpoints)
        parent_dir = Path(output_dir).parent
        run_info_path = parent_dir / "run_info.json"
        if not run_info_path.exists():
            run_info = {
                "model_name": getattr(self.cfg, "model_name", None),
                "target_task": getattr(self.cfg, "target_task", None),
                "tasks": getattr(self.cfg, "tasks", []),
                "seed": getattr(self.cfg, "seed", 42),
            }
            with open(run_info_path, "w") as f:
                json.dump(run_info, f, indent=2)
            print(f"[save_model] wrote run_info.json to {run_info_path}")
    
    def train(self, *args, **kwargs):
        """Override train to cleanup vLLM server after training."""
        try:
            result = super().train(*args, **kwargs)
        finally:
            # Cleanup vLLM server after training completes (or fails)
            if getattr(self.args, 'use_vllm', False):
                self._cleanup_vllm_server()
        return result
    
    def evaluate_hf(self, eval_dataset=None, **kwargs):
        """
        Not implemented.

        Use evaluate_lmeval() instead.
        """
        raise NotImplementedError(
            "evaluate_hf is disabled. Use eval_mode='lmeval' instead."
        )

    def evaluate_lmeval(self, *args, **kwargs):
        """Run lm-eval harness evaluation (external process via tmux)."""
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        
        # Barrier 1: All ranks synchronize before starting
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        # Only rank 0 runs lm-eval, others wait
        if rank != 0:
            print(f"[lm-eval rank {rank}] Waiting for rank 0 to complete evaluation...")
            results = {"results": {}}
        else:
            print(f"[lm-eval rank {rank}] Starting lm-eval evaluation")
            
            # Get checkpoint directory (save model first)
            self.save_model()
            ckpt_dir = self.args.output_dir
            
            # Get lm-eval configuration from cfg
            tasks = getattr(self.cfg, 'lmeval_tasks', 'hendrycks_math')
            batch_size = getattr(self.cfg, 'lmeval_batch_size', 'auto')
            gpu = getattr(self.cfg, 'lmeval_gpu', '0')  # GPU index, not cuda:0
            output_path = os.path.join(ckpt_dir, 'lmeval_results')
            
            # vLLM-specific settings from cfg (with defaults matching lmeval_setup.sh)
            dtype = getattr(self.cfg, 'lmeval_dtype', 'auto')
            tensor_parallel_size = getattr(self.cfg, 'lmeval_tensor_parallel_size', '1')
            gpu_memory_utilization = getattr(self.cfg, 'lmeval_gpu_memory_utilization', '0.8')
            max_model_len = getattr(self.cfg, 'lmeval_max_model_len', '20000')
            
            # Generation settings from cfg (with defaults matching lmeval_setup.sh)
            system_instruction = getattr(
                self.cfg, 'lmeval_system_instruction', 
                r'Please reason step by step, and put your final answer within \boxed{}.'
            )
            gen_kwargs = getattr(
                self.cfg, 'lmeval_gen_kwargs',
                'do_sample=True,temperature=0.6,top_p=0.95,max_gen_toks=20000'
            )
            
            # Path to lm-eval script
            script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "grpo" / "lmeval_server.sh"
            
            if not script_path.exists():
                print(f"[lm-eval rank {rank}] WARNING: Script not found at {script_path}")
                print(f"[lm-eval rank {rank}] Falling back to default evaluation")
                results = super().evaluate(*args, **kwargs)
            else:
                # Check if tmux session exists and kill it
                check_session = subprocess.run(
                    ["tmux", "has-session", "-t", "lmeval-server"],
                    capture_output=True,
                )
                
                if check_session.returncode == 0:
                    print(f"[lm-eval rank {rank}] Killing existing lmeval-server tmux session")
                    subprocess.run(["tmux", "kill-session", "-t", "lmeval-server"])
                    time.sleep(1)
                
                # Create PID file path to track the lm-eval process (in output directory)
                pid_file = Path(output_path) / "lmeval_server.pid"
                pid_file.parent.mkdir(parents=True, exist_ok=True)
                if pid_file.exists():
                    pid_file.unlink()
                
                print(f"[lm-eval rank {rank}] Launching lm-eval in tmux session 'lmeval-server'")
                print(f"[lm-eval rank {rank}] CKPT_DIR={ckpt_dir}, TASKS={tasks}, GPU={gpu}")
                
                # Create log file for debugging
                log_file = Path(output_path) / "lmeval_server.log"
                
                # Build a clean environment for lm-eval (same as vLLM)
                # This prevents distributed training env vars from interfering with lm-eval
                path = os.environ.get("PATH", "")
                ld_lib = os.environ.get("LD_LIBRARY_PATH", "")
                
                # This string will be run inside tmux's shell with a clean environment
                # Run the script in foreground (not background) so tmux tracks it properly
                inner_cmd = (
                    "/usr/bin/env -i "  # wipe env completely
                    f"PATH={shlex.quote(path)} "
                    f"LD_LIBRARY_PATH={shlex.quote(ld_lib)} "
                    f"CKPT_DIR={shlex.quote(ckpt_dir)} "
                    f"TASKS={shlex.quote(tasks)} "
                    f"GPU={gpu} "
                    f"BATCH_SIZE={shlex.quote(str(batch_size))} "
                    f"OUTPUT_PATH={shlex.quote(output_path)} "
                    f"DTYPE={shlex.quote(str(dtype))} "
                    f"TENSOR_PARALLEL_SIZE={shlex.quote(str(tensor_parallel_size))} "
                    f"GPU_MEMORY_UTILIZATION={shlex.quote(str(gpu_memory_utilization))} "
                    f"MAX_MODEL_LEN={shlex.quote(str(max_model_len))} "
                    f"SYSTEM_INSTRUCTION={shlex.quote(system_instruction)} "
                    f"GEN_KWARGS={shlex.quote(gen_kwargs)} "
                    f"bash {shlex.quote(str(script_path))} > {shlex.quote(str(log_file))} 2>&1"
                )
                
                # Create new tmux session and run the script with clean environment
                tmux_cmd = [
                    "tmux", "new-session", "-d", "-s", "lmeval-server",
                    inner_cmd
                ]
                
                proc = subprocess.Popen(tmux_cmd)
                proc.wait()  # Wait for tmux to start
                
                # Give tmux a moment to start the process
                time.sleep(2)
                
                print(f"[lm-eval rank {rank}] lm-eval running in tmux session 'lmeval-server'")
                print(f"[lm-eval rank {rank}] Waiting for evaluation to complete (checking tmux session)...")
                
                # Poll the tmux session to see when it completes
                while True:
                    check = subprocess.run(
                        ["tmux", "has-session", "-t", "lmeval-server"],
                        capture_output=True,
                    )
                    if check.returncode != 0:
                        # Session no longer exists, process completed
                        print(f"[lm-eval rank {rank}] lm-eval process completed")
                        break
                    
                    # Check if there's output in the log to show progress
                    if log_file.exists():
                        try:
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                if lines:
                                    # Print last line if it changed (simple progress indicator)
                                    last_line = lines[-1].strip()
                                    if last_line and not hasattr(self, '_last_log_line') or getattr(self, '_last_log_line', '') != last_line:
                                        print(f"[lm-eval rank {rank}] {last_line}")
                                        self._last_log_line = last_line
                        except:
                            pass
                    
                    time.sleep(10)  # Check every 10 seconds
                
                # Try to load results - lm-eval creates timestamped files in nested directories
                import json
                import glob
                results = {}
                
                # First try the direct path (old lm-eval versions)
                results_file = os.path.join(output_path, "results.json")
                
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    print(f"[lm-eval rank {rank}] Loaded results from {results_file}")
                else:
                    # Search for timestamped results files (new lm-eval versions)
                    pattern = os.path.join(output_path, "**", "results_*.json")
                    json_files = glob.glob(pattern, recursive=True)
                    
                    if json_files:
                        # Use the most recent file
                        latest_file = max(json_files, key=os.path.getmtime)
                        with open(latest_file, 'r') as f:
                            results = json.load(f)
                        print(f"[lm-eval rank {rank}] Loaded results from {latest_file}")
                    else:
                        print(f"[lm-eval rank {rank}] WARNING: No results files found in {output_path}")
                        results = {"results": {}}
        
        # Barrier 2: All ranks synchronize after rank 0 completes (or all ranks if rank 0 had error)
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        return results
    
    def evaluate(self, eval_dataset=None, **kwargs):
        """
        Main evaluate method. Dispatches to either HF or lm-eval based on eval_mode.
        
        Args:
            eval_dataset: Optional evaluation dataset (for HF mode)
            **kwargs: Additional arguments passed to the evaluation method
        
        Config:
            eval_mode: 'lmeval' (default) or 'hf'
                - 'lmeval': Use lm-eval harness (external process)
                - 'hf': Use standard HF trainer evaluation (loss, reward, etc.)
        """
        eval_mode = getattr(self.cfg, 'eval_mode', 'lmeval')
        
        if eval_mode == 'hf':
            return self.evaluate_hf(eval_dataset=eval_dataset, **kwargs)
        elif eval_mode == 'lmeval':
            return self.evaluate_lmeval(**kwargs)
        else:
            raise ValueError(
                f"Unknown eval_mode: {eval_mode}. "
                f"Expected 'hf' or 'lmeval'."
            )