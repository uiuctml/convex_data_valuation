from __future__ import annotations
from typing import Dict, List, Optional, Any
import os
import math
import time
import json
import glob
import shlex
import subprocess
import contextlib
from pathlib import Path
from collections import Counter

import torch.distributed as dist
from trl import SFTTrainer, SFTConfig

import datasets as hf_datasets
import torch

from trainer.multitask_trainer.base import MultiTaskTrainerBase
from data.base_module import TaskSpec, DataModuleConfig
from data.multitask_iterable import MultiTaskOnTheFlyDataset

from data.aya.module import AyaSFTDataModule
from data.aya.formatter import AYA_SFT_FORMATTER
from data.tulu_personas.module import TuluPersonasSFTDataModule
from data.tulu_personas.formatter import TULU_PERSONAS_SFT_FORMATTER
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from modeling.sft.instruction_ft import SFTInstructionModel, SFTInstructionModelConfig


class SFTMultiTaskTrainer(MultiTaskTrainerBase, SFTTrainer):
    """
    Multitask SFT trainer for instruction finetuning.

    - Uses TRL's SFTTrainer under the hood
    - Shares multi-task + token-budget logic with GRPOMultiTaskTrainer
    - Supports two eval modes:
        * eval_mode = "hf"     → HF-style loss/ppl eval
        * eval_mode = "lmeval" → external lm-eval harness via tmux
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
        dm = self._init_or_reuse_data_module(data_module)

        if dm is None:
            task_type = getattr(cfg, "task_type", "aya-sft")

            if task_type == "aya-sft":
                data_module_cls = AyaSFTDataModule
                formatter = AYA_SFT_FORMATTER
                text_field = "messages"
                # Use TRL's DataCollatorForLanguageModeling which expects pre-tokenized data
                pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                collator = DataCollatorForLanguageModeling(
                    pad_token_id=pad_token_id,
                    completion_only_loss=True,
                    padding_free=False,
                )
            elif task_type == "tulu-personas-sft":
                data_module_cls = TuluPersonasSFTDataModule
                formatter = TULU_PERSONAS_SFT_FORMATTER
                text_field = "messages"
                pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                collator = DataCollatorForLanguageModeling(
                    pad_token_id=pad_token_id,
                    completion_only_loss=True,
                    padding_free=False,
                )
            else:
                raise ValueError(
                    f"Unknown SFT task_type '{task_type}'. "
                    f"Supported: 'aya-sft', 'tulu-personas-sft'"
                )

            # Build TaskSpec list with optional few-shot overrides
            specs: List[TaskSpec] = []
            few_overrides = getattr(cfg, "few_shot", None) or {}
            k_shot_global = getattr(cfg, "k_shot", None)

            for t_name in cfg.tasks:
                if t_name in few_overrides:
                    k = few_overrides[t_name]
                elif k_shot_global is not None and t_name != getattr(cfg, "target_task", None):
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
                train_batch_size=cfg.train_batch_size,
                eval_batch_size=cfg.eval_batch_size,
                pin_memory=cfg.pin_memory,
                drop_last=cfg.drop_last,
            )

            dm_kwargs = {
                "tasks": specs,
                "cfg": dm_cfg,
                "formatter": formatter,
                "collator": collator,
                "tokenizer": tok,
            }

            # Pass extra kwargs for TuluPersonas data module
            if task_type == "tulu-personas-sft":
                dm_kwargs["max_samples"] = getattr(cfg, "max_samples", 3000)
                dm_kwargs["sample_seed"] = getattr(cfg, "sample_seed", 42)

            # Pass dataset_split_seed for Aya sub-dataset splitting
            if task_type == "aya-sft":
                dm_kwargs["dataset_split_seed"] = getattr(cfg, "dataset_split_seed", getattr(cfg, "seed", 42))

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
            raise ValueError("SFTMultiTaskTrainer expects `args=SFTConfig(...)` in kwargs.")
        args: SFTConfig = kwargs["args"]

        # Let token-budget logic drive step count
        object.__setattr__(args, "max_steps", planned_max_steps)
        object.__setattr__(args, "remove_unused_columns", False)

        # We will tokenize inside MultiTaskOnTheFlyDataset, so honor the SFT max_seq_length here
        max_seq_length = getattr(args, "max_seq_length", None)

        # ---------------- multitask on-the-fly dataset ----------------
        train_dataset = self._build_sft_multitask_dataset(
            train_loaders=train_loaders,
            mix=mix,
            planned_max_steps=planned_max_steps,
            cfg=cfg,
            tok=tok,
            max_seq_length=max_seq_length,
        )

        self._train_dataset = train_dataset
        self._train_loaders = train_loaders
        self._eval_loaders = eval_loaders
        self._per_task_steps = per_task_steps

        # ---------------- target eval dataset ----------------
        target_eval_ds = None
        if getattr(cfg, "target_task", None) in eval_loaders:
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

        model_config = SFTInstructionModelConfig(
            model_name=cfg.model_name,
            load_in_4bit=getattr(cfg, "load_in_4bit", False),
            load_in_8bit=getattr(cfg, "load_in_8bit", False),
            torch_dtype=getattr(cfg, "torch_dtype", "bfloat16"),
            device_map=getattr(cfg, "device_map", None),
            use_lora=getattr(cfg, "use_lora", True),
            lora_r=getattr(cfg, "lora_r", 64),
            lora_alpha=getattr(cfg, "lora_alpha", 128),
            lora_dropout=getattr(cfg, "lora_dropout", 0.05),
            lora_target_modules=getattr(cfg, "lora_target_modules", lora_target_modules_default),
            lora_task_type=getattr(cfg, "lora_task_type", "CAUSAL_LM"),
            gradient_checkpointing=getattr(cfg, "gradient_checkpointing", False),
        )
        hf_model = SFTInstructionModel(model_config)

        # ---------------- store state ----------------
        self.cfg = cfg
        self.tok = tok
        self.dm = dm

        print(f"[SFT-MT] Tasks: {self.task_names}")
        print(f"[SFT-MT] Mix: {self.mix}")
        print(f"[SFT-MT] Planned max_steps: {planned_max_steps}")

        # ---------------- call base SFTTrainer ----------------
        # NOTE:
        #   - train_dataset is a torch Dataset that already returns input_ids/attention_mask
        #   - target_eval_ds is also pre-tokenized
        #   - we do NOT pass dataset_text_field or data_collator here
        super().__init__(
            model=hf_model,
            processing_class=tok,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=target_eval_ds,
            **{k: v for k, v in kwargs.items() if k != "args"},
        )

    # ------------------------------------------------------------------
    # Override TRL's dataset preparation for our multitask torch Dataset
    # ------------------------------------------------------------------
    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        packing,
        formatting_func,
        dataset_name,
    ):
        """
        Override TRL's prepare_dataset so our MultiTaskOnTheFlyDataset bypasses
        all the HuggingFace Dataset logic.
        """


        # ------------------------------------------------------------------
        # 1. If dataset is *ours* (MultiTaskOnTheFlyDataset), skip processing
        # ------------------------------------------------------------------
        if isinstance(dataset, (torch.utils.data.Dataset, torch.utils.data.IterableDataset)) \
            and not isinstance(dataset, hf_datasets.Dataset) \
            and not isinstance(dataset, hf_datasets.IterableDataset):
            return dataset

        # ------------------------------------------------------------------
        # 2. Otherwise, use TRL's full dataset preparation pipeline
        # ------------------------------------------------------------------
        return super()._prepare_dataset(
            dataset,
            processing_class,
            args,
            packing,
            formatting_func,
            dataset_name,
        )

    # ------------------------------------------------------------------
    # Dataset builder
    # ------------------------------------------------------------------
    def _build_sft_multitask_dataset(
        self,
        train_loaders: Dict,
        mix: Dict[str, float],
        planned_max_steps: int,
        cfg,
        tok,
        max_seq_length: Optional[int],
    ):
        mt_seed = int(getattr(cfg, "seed", 42))

        # Global batch size: per_device * world_size * grad_accum
        if getattr(cfg, "per_device_train_batch_size", None) is not None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            global_bs = (
                cfg.per_device_train_batch_size
                * world_size
                * getattr(cfg, "grad_accum_steps", 1)
            )
        else:
            global_bs = getattr(cfg, "train_batch_size", 1)

        virtual_length = planned_max_steps * global_bs

        return MultiTaskOnTheFlyDataset(
            train_loaders=train_loaders,
            mix=mix,
            length=virtual_length,
            tokenizer=tok,
            seed=mt_seed,
            max_seq_length=max_seq_length,
        )

    # ------------------------------------------------------------------
    # Save hook
    # ------------------------------------------------------------------
    def save_model(self, output_dir: Optional[str] = None, **kwargs):
        if output_dir is None:
            output_dir = self.args.output_dir

        # The model is wrapped in SFTInstructionModel, so we need to access .model attribute
        if hasattr(self.model, 'model'):
            # Save the inner model (the actual transformer)
            self.model.model.save_pretrained(
                output_dir,
                safe_serialization=False,
            )
        else:
            # Fallback to saving self.model directly
            self.model.save_pretrained(
                output_dir,
                safe_serialization=False,
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        print(f"[SFT save_model] Model saved to {output_dir} (safe_serialization=False)")

    def evaluate_hf(self, eval_dataset=None, **kwargs):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if eval_dataset is None:
            return {}

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        
        losses: list[float] = []
        f1_sum_local = 0.0
        n_ex_local = 0.0

        ctx = getattr(self, "maybe_activation_offload_context", contextlib.nullcontext())

        with torch.no_grad(), ctx:
            for step, inputs in enumerate(eval_dataloader):
                for k, v in list(inputs.items()):
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.accelerator.device)

                # 1) Get outputs. We need the logits but we want to discard them FAST.
                loss, outputs = self.compute_loss(self.model, inputs, return_outputs=True)
                
                # 2) Extract only what we need for F1 (the predictions)
                # This reduces [B, T, V] -> [B, T] immediately
                logits = outputs.logits
                
                # Align shift: Logits at i predict Label at i+1
                shift_logits = logits[:, :-1, :]
                shift_labels = inputs.get("labels")[:, 1:]
                shift_mask = (shift_labels != -100)

                # Calculate predictions and move to CPU if F1 is slow/heavy
                # or keep on GPU if it fits. argmax is the memory-saving step.
                pred_ids = shift_logits.argmax(dim=-1) 

                # 3) AGGRESSIVE CLEANUP
                # We have pred_ids now, we don't need the massive logits anymore.
                del outputs
                del logits
                del shift_logits

                # Standard loss tracking
                loss_gathered = self.accelerator.gather_for_metrics(loss.detach())
                losses.append(loss_gathered.mean().item())

                # 4) F1 Calculation
                if shift_mask.any():
                    for i in range(shift_labels.size(0)):
                        m = shift_mask[i]
                        if not m.any(): continue
                        
                        gold = shift_labels[i][m].tolist()
                        pred = pred_ids[i][m].tolist()
                        
                        gold_c, pred_c = Counter(gold), Counter(pred)
                        num_same = sum((gold_c & pred_c).values())
                        
                        prec = num_same / len(pred) if len(pred) > 0 else 0
                        rec = num_same / len(gold) if len(gold) > 0 else 0
                        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                        
                        f1_sum_local += f1
                        n_ex_local += 1.0

        # --- Final Aggregation ---
        metrics = {}
        if losses:
            mean_loss = sum(losses) / len(losses)
            metrics["eval_loss"] = mean_loss
            metrics["eval_ppl"] = math.exp(min(mean_loss, 20))

        f1_data = torch.tensor([f1_sum_local, n_ex_local], device=self.accelerator.device)
        f1_data_g = self.accelerator.gather_for_metrics(f1_data)
        
        if f1_data_g.dim() > 1:
            f1_data_g = f1_data_g.sum(dim=0)
        
        f1_total, ex_total = f1_data_g.tolist()
        if ex_total > 0:
            metrics["eval_token_f1"] = f1_total / ex_total

        if self.args.local_rank in (-1, 0):
            print(f"[SFT evaluate_hf] Metrics: {metrics}")

        self.log(metrics)
        return metrics

    # ------------------------------------------------------------------
    # lm-eval evaluation (copied from GRPO version, slightly simplified)
    # ------------------------------------------------------------------
    def evaluate_lmeval(self, *args, **kwargs):
        """Run lm-eval harness evaluation (external process via tmux)."""
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            print(f"[lm-eval rank {rank}] Waiting for rank 0 to complete evaluation...")
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1 and dist.is_initialized():
                dist.barrier()
            return {"results": {}}

        print(f"[lm-eval rank {rank}] Starting lm-eval evaluation")

        # Save current checkpoint first
        self.save_model()
        ckpt_dir = self.args.output_dir

        # Get lm-eval configuration from cfg
        tasks = getattr(self.cfg, "lmeval_tasks", "hendrycks_math")
        batch_size = getattr(self.cfg, "lmeval_batch_size", "auto")
        gpu = getattr(self.cfg, "lmeval_gpu", "0")  # GPU index (e.g., "0")
        output_path = os.path.join(ckpt_dir, "lmeval_results")
        
        # Create unique session name: GPU ID + short hash of ckpt_dir
        # This prevents collisions when multiple experiments share the same GPU
        import hashlib
        _ckpt_hash = hashlib.md5(ckpt_dir.encode()).hexdigest()[:6]
        session_name = f"lmeval-gpu{gpu}-{_ckpt_hash}"

        # Path to lm-eval script
        script_path = (
            Path(__file__).parent.parent.parent.parent
            / "scripts"
            / "sft"
            / "lmeval_server.sh"
        )

        if not script_path.exists():
            print(f"[lm-eval rank {rank}] WARNING: Script not found at {script_path}")
            print(f"[lm-eval rank {rank}] Falling back to HF evaluation")
            return self.evaluate_hf(*args, **kwargs)

        # Kill existing tmux session if any
        check_session = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True,
        )
        if check_session.returncode == 0:
            print(f"[lm-eval rank {rank}] Killing existing {session_name} tmux session")
            subprocess.run(["tmux", "kill-session", "-t", session_name])
            time.sleep(1)

        # PID/log files
        pid_file = Path(output_path) / "lmeval_server.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        if pid_file.exists():
            pid_file.unlink()

        log_file = Path(output_path) / "lmeval_server.log"

        print(f"[lm-eval rank {rank}] Launching lm-eval in tmux session '{session_name}'")
        print(f"[lm-eval rank {rank}] CKPT_DIR={ckpt_dir}, TASKS={tasks}, GPU={gpu}")

        # Build clean env
        path = os.environ.get("PATH", "")
        ld_lib = os.environ.get("LD_LIBRARY_PATH", "")

        inner_cmd = (
            "/usr/bin/env -i "
            f"PATH={shlex.quote(path)} "
            f"LD_LIBRARY_PATH={shlex.quote(ld_lib)} "
            f"CKPT_DIR={shlex.quote(ckpt_dir)} "
            f"TASKS={shlex.quote(tasks)} "
            f"GPU={gpu} "
            f"BATCH_SIZE={shlex.quote(str(batch_size))} "
            f"OUTPUT_PATH={shlex.quote(output_path)} "
            f"bash {shlex.quote(str(script_path))} > {shlex.quote(str(log_file))} 2>&1"
        )

        tmux_cmd = [
            "tmux", "new-session", "-d", "-s", session_name,
            inner_cmd,
        ]

        proc = subprocess.Popen(tmux_cmd)
        proc.wait()
        time.sleep(2)

        print(f"[lm-eval rank {rank}] lm-eval running in tmux session '{session_name}'")
        print(f"[lm-eval rank {rank}] Waiting for evaluation to complete (checking tmux session)...")

        while True:
            check = subprocess.run(
                ["tmux", "has-session", "-t", session_name],
                capture_output=True,
            )
            if check.returncode != 0:
                print(f"[lm-eval rank {rank}] lm-eval process completed")
                break

            if log_file.exists():
                try:
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if last_line and getattr(self, "_last_log_line", "") != last_line:
                                print(f"[lm-eval rank {rank}] {last_line}")
                                self._last_log_line = last_line
                except Exception:
                    pass

            time.sleep(10)

        # Load results
        results = {}
        results_file = os.path.join(output_path, "results.json")

        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results = json.load(f)
            print(f"[lm-eval rank {rank}] Loaded results from {results_file}")
        else:
            pattern = os.path.join(output_path, "**", "results_*.json")
            json_files = glob.glob(pattern, recursive=True)
            if json_files:
                latest_file = max(json_files, key=os.path.getmtime)
                with open(latest_file, "r") as f:
                    results = json.load(f)
                print(f"[lm-eval rank {rank}] Loaded results from {latest_file}")
            else:
                print(f"[lm-eval rank {rank}] WARNING: No results files found in {output_path}")

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        return results

    # ------------------------------------------------------------------
    # Main evaluate dispatcher
    # ------------------------------------------------------------------
    def evaluate(self, eval_dataset=None, **kwargs):
        """
        Dispatch between HF evaluation and lm-eval harness.

        cfg.eval_mode:
            - "hf" (default)     → use evaluate_hf()
            - "lmeval"           → use evaluate_lmeval()
        """
        eval_mode = getattr(self.cfg, "eval_mode", "hf")

        if eval_mode == "hf":
            return self.evaluate_hf(eval_dataset=eval_dataset, **kwargs)
        elif eval_mode == "lmeval":
            return self.evaluate_lmeval(**kwargs)
        else:
            raise ValueError(
                f"Unknown eval_mode: {eval_mode}. Expected 'hf' or 'lmeval'."
            )