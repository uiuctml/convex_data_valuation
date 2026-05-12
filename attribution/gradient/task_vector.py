from __future__ import annotations
"""
Task-vector dataset attribution that reuses the existing multitask Trainer
via TrainerModelFactory.

Design:
  - Single-task fine-tunes for target and each aux.
  - Per-task token budgets derived from the original global_token_budget
    and (target_weight, aux_weight_total) using the same make_mix_weights
    logic as the multitask trainer.
  - Reuses:
      * factory.model_init_fn()          → base model
      * factory.trainable_param_filter   → which params are trainable
      * factory.trainer.cfg              → optimizer/schedule/token budget logic
      * factory.data_module              → datasets / loaders
      * factory.make_single_task_trainer → to avoid rebuilding data
  - ResultStore and AttributionRunInfo are passed from the outside.
"""

from typing import Dict, Optional
import tempfile

import torch
from torch import nn

from attribution.gradient.base import GradientAttributionBase
from attribution.save import ResultStore, AttributionRunInfo
from trainer.multitask_trainer.allocate_tokens import make_mix_weights

class TaskVectorAttribution(GradientAttributionBase):
    """
    Task-vector similarity attribution.

    For each task t:

      θ_t = params after fine-tuning on task t only
      v_t = θ_t - θ_base

    Then attribution scores for aux tasks are based on similarity(v_target, v_aux).

    This implementation:
      - Uses TrainerModelFactory to reuse an existing multitask Trainer.
      - Uses factory.make_single_task_trainer(...) so we don't rebuild datasets.
      - Allocates per-task token budgets consistent with the original MTL mix.
    """

    # ---------- Internal helpers ----------

    def _finetune_and_get_model(
        self,
        base_model: nn.Module,
        task_name: str,
        tmp_prefix: str,
        token_budget_override: Optional[int],
        debug_mode: bool = True,
    ) -> nn.Module:
        """
        Fine-tune a copy of base_model on a single task and return the nn.Module.

        We:
          - Ask the factory for a single-task trainer (reusing cfg + data_module).
          - Override global_token_budget for this run (if provided).
          - Load base_model's weights into trainer.model so all runs start
            from the same θ_base.
        
        Args:
            debug_mode: If True, skip training and return base model with small random noise
        """
        if debug_mode:
            print(f"[TaskVectorAttribution] DEBUG MODE: Skipping fine-tuning for {task_name}")
            # Return a copy of base model with small random perturbations to simulate fine-tuning
            import copy
            model = copy.deepcopy(base_model)
            # Add small random noise to parameters to simulate fine-tuning differences
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.add_(torch.randn_like(param) * 1e-4)
            return model.to(self.device)
        
        with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir:
            trainer = self.factory.make_single_task_trainer(
                task_name=task_name,
                output_dir=tmpdir,
                token_budget_override=token_budget_override,
            )

            # Ensure identical initialization across all runs
            base_sd = base_model.state_dict()
            target_sd = trainer.model.state_dict()

            remapped = {}

            for k, v in base_sd.items():
                new_k = k

                # HF LM: "model.xxx"  -> MathRLModel: "model.model.xxx"
                if k.startswith("model."):
                    suffix = k[len("model."):]
                    candidate = f"model.model.{suffix}"
                    if candidate in target_sd:
                        new_k = candidate

                # HF LM: "lm_head.weight" -> MathRLModel: "model.lm_head.weight"
                elif k == "lm_head.weight":
                    candidate = "model.lm_head.weight"
                    if candidate in target_sd:
                        new_k = candidate

                # Only keep if the remapped key actually exists in target
                if new_k in target_sd:
                    remapped[new_k] = v
                else:
                    # optional: print for debugging
                    # print(f"[state_dict remap] skipping {k} -> {new_k} (not in target)")
                    pass

            trainer.train()

            model = trainer.model
            if hasattr(model, "module"):  # DDP or DataParallel
                model = model.module
            return model.to(self.device)

    def _task_vector(
        self,
        pretrained: nn.Module,
        finetuned: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task vector as a mapping name -> delta(param) over trainable params.
        """
        pre_params = dict(self._named_trainable_params(pretrained))
        fin_params = dict(self._named_trainable_params(finetuned))

        tv: Dict[str, torch.Tensor] = {}
        for name in pre_params.keys():
            tv[name] = (fin_params[name] - pre_params[name]).detach().cpu().clone()
        return tv

    # ---------- Public API ----------

    def score_auxiliary_datasets(
        self,
        *,
        result_store: ResultStore,
        run_info: AttributionRunInfo,
        similarity: str = "cos",
        save_artifacts: bool = True,
        debug_mode: bool = False,
    ) -> Dict[str, float]:
        """
        Compute TaskVector similarity scores for auxiliary tasks.

        Automatically infers target_task_name, aux_task_names, and total_token_budget
        from the trainer factory's configuration.

        Args:
            result_store:
                ResultStore to use for saving artifacts and scores.

            run_info:
                AttributionRunInfo, constructed by the caller; we do not
                modify or recreate it here.

            similarity:
                Similarity metric passed to similarity_state_dict (e.g., "cos").

            save_artifacts:
                If True, save task vectors as artifacts (default: True).
            
            debug_mode:
                If True, skip fine-tuning and use dummy task vectors (default: False).

        Returns:
            Dict[str, float]: mapping aux_task -> similarity(v_target, v_aux).
        """
        import time
        
        # Start timing for score_aux_datasets
        start_time = time.time()
        
        cfg = self.factory.trainer.cfg

        # ------------------------------------------------------------------
        # 0) Infer target_task_name, aux_task_names, and total_token_budget
        #    from cfg
        # ------------------------------------------------------------------
        target_task_name = getattr(cfg, "target_task", None)
        if target_task_name is None:
            raise ValueError("cfg.target_task is not set in the trainer factory.")
        
        all_tasks_cfg = list(getattr(cfg, "tasks", []))
        # Infer aux tasks as all tasks except the target
        aux_task_names = [t for t in all_tasks_cfg if t != target_task_name]
        if not aux_task_names:
            raise ValueError(
                f"No auxiliary tasks found. cfg.tasks={all_tasks_cfg}, "
                f"target_task_name={target_task_name}"
            )

        # ------------------------------------------------------------------
        # 1) Determine total token budget for TaskVector runs
        # ------------------------------------------------------------------
        cfg_budget = getattr(cfg, "global_token_budget", None)
        if cfg_budget is None:
            raise ValueError("cfg.global_token_budget is not set in the trainer factory.")
        B_total = int(cfg_budget)

        # ------------------------------------------------------------------
        # 2) Compute original multitask mix over *all* tasks, then induce
        #    per-task budgets consistent with that mix.
        # ------------------------------------------------------------------
        # All tasks in original MTL; we only care about target + aux we score,
        # but mixing should respect the full task set.
        cfg_target = target_task_name
        T = float(getattr(cfg, "target_weight", 1.0))
        A = float(getattr(cfg, "aux_weight_total", 0.0))

        full_mix = make_mix_weights(
            tasks=all_tasks_cfg,
            target=cfg_target,
            target_w=T,
            aux_total_w=A,
        )

        # Per-task budgets (ensure at least 1 token's worth if B_total > 0 and mix[t] > 0)
        per_task_budget: Dict[str, int] = {}
        interested_tasks = {target_task_name, *aux_task_names}
        for t in interested_tasks:
            frac = float(full_mix.get(t, 0.0))
            if B_total > 0 and frac > 0.0:
                per_task_budget[t] = max(1, int(B_total * frac))
            else:
                per_task_budget[t] = 0

        # ------------------------------------------------------------------
        # 3) Prepare run directory and base model - only on rank 0
        # ------------------------------------------------------------------
        # Check if we're in distributed mode
        trainer = getattr(self.factory, "trainer", None)
        accelerator = getattr(trainer, "accelerator", None)
        rank = accelerator.process_index if accelerator is not None else 0
        
        # Only rank 0 creates run directory and saves artifacts
        if rank == 0:
            run_dir = result_store.new_run_dir(
                method_name=run_info.method_name,
                target_task=run_info.target_task,
            )
            result_store.save_run_info(run_dir, run_info)
        else:
            run_dir = None  # Non-zero ranks don't need run_dir
            print(f"[TaskVectorAttribution rank {rank}] Skipping run directory creation (only rank 0 saves)")

        base_ref = self.model_init_fn().to(self.device)
        base_ref.eval()

        # ------------------------------------------------------------------
        # 4) Target-only fine-tune → target task vector
        # ------------------------------------------------------------------
        B_target_run = per_task_budget.get(target_task_name, 0)
        target_model = self._finetune_and_get_model(
            base_model=base_ref,
            task_name=target_task_name,
            tmp_prefix="taskvec_target_",
            token_budget_override=B_target_run,
            debug_mode=debug_mode,
        )
        tv_target = self._task_vector(pretrained=base_ref, finetuned=target_model)
        if save_artifacts and rank == 0:
            result_store.save_artifact(run_dir, "target_task_vector", tv_target)
        
        # Free target model memory
        del target_model
        torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # 5) Aux-only fine-tunes → aux task vectors and similarity scores
        # ------------------------------------------------------------------
        scores: Dict[str, float] = {}
        for aux_name in aux_task_names:
            B_aux_run = per_task_budget.get(aux_name, 0)

            aux_model = self._finetune_and_get_model(
                base_model=base_ref,
                task_name=aux_name,
                tmp_prefix=f"taskvec_aux_{aux_name}_",
                token_budget_override=B_aux_run,
                debug_mode=debug_mode,
            )
            tv_aux = self._task_vector(pretrained=base_ref, finetuned=aux_model)
            if save_artifacts and rank == 0:
                result_store.save_artifact(run_dir, f"aux_{aux_name}_task_vector", tv_aux)

            # Free aux model memory before computing similarity
            del aux_model
            torch.cuda.empty_cache()

            score = self.similarity_state_dict(tv_target, tv_aux, kind=similarity)
            scores[aux_name] = score
            print(f"[TaskVectorAttribution] Similarity({target_task_name}, {aux_name}) = {score:.6f}")

        # ------------------------------------------------------------------
        # 6) Clean up and save scores - only on rank 0
        # ------------------------------------------------------------------
        # Free base model memory
        del base_ref
        torch.cuda.empty_cache()
        
        # Record timing
        elapsed_time = time.time() - start_time
        
        if rank == 0:
            # Save timing information as artifact
            if save_artifacts:
                timing_info = {
                    "score_aux_datasets_seconds": elapsed_time,
                }
                result_store.save_artifact(run_dir, "timing", timing_info)
            
            result_store.save_scores(run_dir, scores)
        else:
            print(f"[TaskVectorAttribution rank {rank}] Skipping score save (only rank 0 saves)")
        
        return scores