from __future__ import annotations

"""
One-step gradient dataset attribution.

Computes the average gradient on the target dataset, then scores each auxiliary
(pre-truncated to k-shot *outside* this class) by cosine or dot similarity.

Persistence:
    Pass a ResultStore via `result_store=` to save scores + averaged gradients.
    If omitted, nothing is written to disk.
"""

from typing import Callable, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler

from attribution.gradient.base import GradientAttributionBase
from attribution.save import ResultStore, AttributionRunInfo

from collections.abc import Mapping

from tqdm import tqdm

Batch = Mapping[str, torch.Tensor]
LossFn = Callable[[nn.Module, Batch], torch.Tensor]


class OneStepGradientAttribution(GradientAttributionBase):
    """Gradient-similarity attribution with a single avg-gradient routine for all datasets."""

    # ---- Internal helper: grad on batch ----
    def _accumulate_grad_on_batch(
        self,
        model: nn.Module,
        batch,
        loss_fn: LossFn,
        param_names_sorted: list,  # Only extract grads for these params
    ) -> Tuple[dict, int]:
        trainer = self.factory.trainer
        accelerator = getattr(trainer, "accelerator", None)

        model.train()

        # IMPORTANT: free any existing grads instead of just zeroing
        for p in model.parameters():
            p.grad = None

        loss = loss_fn(model, batch)  # calls trainer.compute_loss(...)

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

        # Copy grads to CPU for accumulation - only for filtered parameters
        grad_cpu = {}
        param_dict = dict(model.named_parameters())
        for name in param_names_sorted:
            p = param_dict[name]
            if p.grad is not None:
                grad_cpu[name] = p.grad.detach().cpu()
            else:
                # Parameter had no gradient (shouldn't happen for trainable params, but be safe)
                grad_cpu[name] = torch.zeros_like(p.data, device='cpu')

        # Infer batch size for weighting
        # For GRPO: batch is list[dict] of unique prompts (before num_generations duplication)
        # The loss_fn handles duplication internally, so n = number of unique prompts
        if isinstance(batch, list):
            # GRPO: list of unique prompts - each contributes one gradient
            n = len(batch)
        elif isinstance(batch, Mapping) and "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
            n = int(batch["input_ids"].size(0))
        else:
            n = 1

        # FREE gradients again so they don’t accumulate per-batch
        for p in model.parameters():
            p.grad = None

        # Drop references and flush CUDA cache
        del loss
        torch.cuda.empty_cache()

        return grad_cpu, n
    

    def _move_to_device(self, obj, device):
        def _move(x):
            return x.to(device) if hasattr(x, "to") and callable(x.to) else x
        if isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            # don't convert lists—just recurse and leave them on CPU
            return type(obj)(self._move_to_device(v, device) for v in obj)
        return _move(obj)

    # ---- Helper to create non-distributed loader from existing loader ----
    def _make_nondistributed_loader(self, loader: DataLoader) -> DataLoader:
        """
        Create a non-distributed DataLoader from an existing (possibly distributed) loader.
        This ensures ALL ranks iterate over ALL data, which is necessary for proper
        gradient computation where we manually shard each batch.
        """
        # Get the underlying dataset
        dataset = loader.dataset
        
        # Extract loader parameters
        batch_size = loader.batch_size
        collate_fn = loader.collate_fn
        
        # Create new loader WITHOUT distributed sampler
        # Use sequential sampler so all ranks see the same order
        new_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collate_fn,
            num_workers=0,  # Use 0 workers to avoid multiprocessing issues in distributed setting
            pin_memory=False,  # Disable to avoid CUDA issues
            drop_last=False,
        )
        
        return new_loader

    def _make_distributed_loader(self, loader: DataLoader, rank: int, world_size: int) -> DataLoader:
        """
        Create a distributed DataLoader from an existing loader.
        Each rank will see 1/world_size of the data.
        """
        dataset = loader.dataset
        batch_size = loader.batch_size
        collate_fn = loader.collate_fn
        
        # Create DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # Don't shuffle for reproducibility in attribution
            drop_last=False,
        )
        
        new_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        
        return new_loader

    # ---- Unified average-gradient helper ----
    @torch.inference_mode(False)
    def avg_gradient(
        self,
        *,
        loader: DataLoader,
        loss_fn: LossFn,
        max_batches: Optional[int] = None,
        debug_mode: bool = False,
    ) -> dict:
        """
        Average gradient state_dict over `loader`, with proper multi-GPU handling.

        Strategy for multi-GPU with vLLM server mode:
        - Each rank iterates through its OWN shard of data (via DistributedSampler)
        - All ranks call loss_fn on their local batch TOGETHER (synchronized)
        - Inside loss_fn → _prepare_inputs: prompts are GATHERED from all ranks,
          main process calls vLLM, results are BROADCAST back
        - So each "step" processes world_size batches worth of prompts
        - Gradients are accumulated locally, then SUM-reduced across ranks at the end
        
        Args:
            debug_mode: If True, skip gradient computation and return dummy gradients
        """
        # Use trainer's accelerator if present
        trainer = getattr(self.factory, "trainer", None)
        accelerator = getattr(trainer, "accelerator", None)
        
        # Get rank/world info
        rank = accelerator.process_index if accelerator is not None else 0
        world_size = accelerator.num_processes if accelerator is not None else 1
        device = accelerator.device if accelerator is not None else self.device
        
        # Decide distribution strategy based on dataset size
        original_len = len(loader.dataset)
        use_distributed = world_size > 1 and original_len >= world_size
        
        # For small datasets (< world_size), don't distribute - all ranks see all data
        # but only rank 0 accumulates gradients. All ranks still call loss_fn for vLLM sync.
        if world_size > 1:
            if use_distributed:
                # Normal case: distribute data across ranks
                loader = self._make_distributed_loader(loader, rank, world_size)
                print(f"[one-step-grad][rank {rank}/{world_size}] Distributed mode: "
                      f"{original_len} samples -> {len(loader)} batches/rank")
            else:
                # Small dataset: all ranks see all data, only rank 0 accumulates
                loader = self._make_nondistributed_loader(loader)
                print(f"[one-step-grad][rank {rank}/{world_size}] Small dataset mode ({original_len} < {world_size}): "
                      f"all ranks see all {len(loader)} batches, only rank 0 accumulates gradients")
        
        # Calculate how many batches each rank will iterate
        num_batches_per_rank = len(loader)
        
        # Get num_generations from trainer (for GRPO) or factory
        num_generations = getattr(self.factory, "num_generations", 1)
        if num_generations == 1 and trainer is not None:
            num_generations = getattr(trainer, "num_generations", 1)
        
        # For small dataset mode, only 1 rank's worth of prompts go to vLLM per step
        # (all ranks have same batch, but gather still collects world_size copies)
        if use_distributed:
            prompts_per_step = world_size * loader.batch_size
        else:
            # Small dataset: all ranks have SAME batch, so gather gets world_size copies of same prompt
            # vLLM deduplicates, so effectively 1 * batch_size unique prompts per step
            prompts_per_step = loader.batch_size  # only unique prompts
        
        print(f"[one-step-grad][rank {rank}/{world_size}] Starting avg_gradient")
        print(f"[one-step-grad][rank {rank}/{world_size}] Loader: {num_batches_per_rank} batches/rank, "
              f"{original_len} total samples")
        print(f"[one-step-grad][rank {rank}/{world_size}] batch_size={loader.batch_size}, num_generations={num_generations}, "
              f"unique prompts/step={prompts_per_step}, vLLM gens/step={prompts_per_step * num_generations}")
        
        # Get model and determine filtered parameter names first (needed for both debug and real mode)
        model = self.model_init_fn()
        if accelerator is None:
            model = model.to(self.device)
        
        # Get parameter names upfront using trainable_param_filter (all ranks have same model structure)
        # This ensures we only compute gradients for the parameters that actually need them (e.g., LoRA params)
        param_filter = self.trainable_param_filter
        param_names_sorted = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                # Apply trainable_param_filter if available
                if param_filter is None or param_filter(name, p):
                    param_names_sorted.append(name)
        param_names_sorted = sorted(param_names_sorted)
        
        print(f"[one-step-grad][rank {rank}] Model has {len(param_names_sorted)} trainable parameters (filtered)")
        
        # DEBUG MODE: Return dummy gradients without computation
        if debug_mode:
            print(f"[one-step-grad][rank {rank}] DEBUG MODE: Skipping gradient computation, returning dummy gradients")
            # Use cfg seed so all ranks get the SAME dummy gradients
            # This ensures consistent scores across ranks for debugging
            cfg = getattr(self.factory.trainer, "cfg", None)
            seed = getattr(cfg, "seed", 42) if cfg is not None else 42
            rng = torch.Generator().manual_seed(seed)
            dummy_grads = {}
            param_dict = dict(model.named_parameters())
            for name in param_names_sorted:
                dummy_grads[name] = torch.randn(param_dict[name].shape, generator=rng, device='cpu') * 1e-4
            print(f"[one-step-grad][rank {rank}] Generated {len(dummy_grads)} dummy gradient tensors (filtered, seed={seed})")
            return dummy_grads
        
        # Initialize accumulators with zeros (ensures all ranks have same keys)
        # Only for parameters that pass the filter
        accum = {}
        param_dict = dict(model.named_parameters())
        for name in param_names_sorted:
            accum[name] = torch.zeros_like(param_dict[name].data, device='cpu')
        
        total_n = 0  # samples accumulated on THIS rank
        batches_processed = 0  # batches this rank processed

        # tqdm bar; show only on local main process
        # Each rank iterates through its own shard (len(loader) batches)
        # But each "step" is synchronized across all ranks, processing world_size batches total
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"[rank {rank}] Computing gradients",
            disable=bool(accelerator and not accelerator.is_local_main_process),
        )

        for i, batch in pbar:
            if max_batches is not None and i >= max_batches:
                # All ranks must break at the same point to stay synchronized
                break
            
            # Determine batch size (unique samples, not duplicated by num_generations)
            # For GRPO: batch is list[dict], each dict is one unique prompt
            if isinstance(batch, list):
                # GRPO: list of unique prompts
                batch_size = len(batch)
            elif isinstance(batch, dict) and "input_ids" in batch:
                batch_size = batch["input_ids"].size(0)
            else:
                batch_size = 1
            
            # DEBUG: Log first batch this rank processes
            if batches_processed == 0:
                print(f"[DEBUG][rank {rank}] First batch (i={i}): type={type(batch).__name__}, "
                      f"local_batch_size={batch_size}, use_distributed={use_distributed}")
            
            # Move dict batches to device
            if isinstance(batch, dict):
                batch = {k: self._move_to_device(v, device) for k, v in batch.items()}
            
            try:
                # ALL ranks must call loss_fn together for vLLM gather/broadcast synchronization
                # In distributed mode: each rank has different batch
                # In small dataset mode: all ranks have SAME batch
                grad_dict, n = self._accumulate_grad_on_batch(
                    model=model,
                    batch=batch,
                    loss_fn=loss_fn,
                    param_names_sorted=param_names_sorted,
                )
                
                # Accumulation strategy depends on mode:
                # - Distributed mode: each rank accumulates its own gradients, then reduce at end
                # - Small dataset mode: only rank 0 accumulates (avoid double-counting)
                should_accumulate = use_distributed or (rank == 0)
                
                if should_accumulate:
                    for k in accum.keys():
                        if k in grad_dict:
                            accum[k] = accum[k] + grad_dict[k] * n
                    total_n += n
                    batches_processed += 1
                
            except Exception as e:
                print(f"[one-step-grad][rank {rank}] ERROR on batch {i}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next batch - don't break the loop
            
            # Update progress bar
            if use_distributed:
                # Distributed: each step processes world_size batches (one per rank)
                global_n_estimate = (i + 1) * world_size * batch_size
            else:
                # Small dataset: each step processes 1 batch (same across all ranks)
                global_n_estimate = (i + 1) * batch_size
            
            if isinstance(batch, list) and batch and isinstance(batch[0], Mapping) and "task" in batch[0]:
                tasks = sorted({ex["task"] for ex in batch})
                pbar.set_postfix_str(f"local_n={total_n}, global_n≈{global_n_estimate}, tasks={','.join(tasks)}")
            else:
                pbar.set_postfix_str(f"local_n={total_n}, global_n≈{global_n_estimate}")

        print(f"[one-step-grad][rank {rank}] Local accumulation done: "
              f"{batches_processed} batches processed on this rank, "
              f"{total_n} local samples")

        # -------- cross-rank reduction --------
        if accelerator is not None and world_size > 1:
            if use_distributed:
                # Distributed mode: sum gradients across ranks
                print(f"[one-step-grad][rank {rank}] Starting cross-rank reduction...")
                
                for key in param_names_sorted:
                    accum[key] = accelerator.reduce(accum[key].to(device), reduction="sum").cpu()
                
                # Reduce sample counts
                total_n_tensor = torch.tensor(float(total_n), device=device)
                total_n_tensor = accelerator.reduce(total_n_tensor, reduction="sum")
                total_n = int(total_n_tensor.item())
                
                print(f"[one-step-grad][rank {rank}] Reduction done. Total samples across all ranks: {total_n}")
            else:
                # Small dataset mode: only rank 0 has gradients, broadcast to other ranks
                print(f"[one-step-grad][rank {rank}] Small dataset mode: broadcasting gradients from rank 0...")
                
                for key in param_names_sorted:
                    accum[key] = accelerator.reduce(accum[key].to(device), reduction="sum").cpu()
                
                # Broadcast total_n from rank 0
                total_n_tensor = torch.tensor(float(total_n), device=device)
                total_n_tensor = accelerator.reduce(total_n_tensor, reduction="sum")
                total_n = int(total_n_tensor.item())
                
                print(f"[one-step-grad][rank {rank}] Broadcast done. Total samples: {total_n}")

        if total_n == 0:
            print(f"[one-step-grad][rank {rank}] WARNING: total_n=0, returning zero gradients")
            return accum

        # Compute average
        avg_grad = {k: v / total_n for k, v in accum.items()}
        print(f"[one-step-grad][rank {rank}] Returning averaged gradients over {total_n} samples")
        
        return avg_grad

    # ---- Main scoring API ----
    def score_auxiliary_datasets(
        self,
        *,
        similarity: str = "cos",
        target_max_batches: Optional[int] = None,
        # ---- optional persistence hooks ----
        result_store: Optional[ResultStore] = None,
        run_info: Optional[AttributionRunInfo] = None,
        save_artifacts: bool = True,
    ) -> Dict[str, float]:
        """Compute and (optionally) persist attribution scores.

        Args:
            similarity: 'cos' (default) or 'dot'.
            target_max_batches: cap number of target batches to average over.

            result_store: if provided, results will be saved using this store.
            run_info: optional AttributionRunInfo to override/extend metadata.
            save_artifacts: if True, save averaged gradients as artifacts.

        Returns:
            Dict[str, float]: aux_name -> similarity score.
        """
        import time
        
        # Start timing for score_aux_datasets
        start_time = time.time()
        
        # Extract from trainer_factory
        target_loader = self.factory.target_loader
        aux_loaders = self.factory.aux_loaders
        loss_fn = self.factory.loss_fn
        target_task_name = self.factory.target_task
        
        # Compute target avg gradient (as state_dict)
        target_avg_grad = self.avg_gradient(
            loader=target_loader, loss_fn=loss_fn, max_batches=target_max_batches, debug_mode=False
        )

        # Compute aux avg gradients and scores
        aux_avg_grads: Dict[str, dict] = {}
        scores: Dict[str, float] = {}
        for name, loader in aux_loaders.items():
            g = self.avg_gradient(loader=loader, loss_fn=loss_fn, debug_mode=False)  # loader already k-shot
            aux_avg_grads[name] = g
            scores[name] = self.similarity_state_dict(target_avg_grad, g, kind=similarity)
        
        # Check rank for printing
        trainer = getattr(self.factory, "trainer", None)
        accelerator = getattr(trainer, "accelerator", None)
        rank = accelerator.process_index if accelerator is not None else 0
        
        # Print nicely - only on rank 0 since all ranks have the same reduced gradients
        if rank == 0:
            print("Target:", target_task_name)
            for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
                print(f"{k}: {v:.6f}")

        # Optional persistence - only on rank 0 to avoid duplicate saves
        if result_store is not None:
            # Check if we're in distributed mode
            trainer = getattr(self.factory, "trainer", None)
            accelerator = getattr(trainer, "accelerator", None)
            rank = accelerator.process_index if accelerator is not None else 0
            
            if rank == 0:
                if run_info is None:
                    raise ValueError("run_info must be provided when result_store is specified")
                
                method_name = "one_step"
                tgt_name = target_task_name

                # Build artifacts to save (averaged gradients)
                artifacts = {}
                if save_artifacts:
                    # Record timing
                    elapsed_time = time.time() - start_time
                    
                    artifacts = {
                        "target_avg_grad": target_avg_grad,
                        "aux_avg_grads": aux_avg_grads,  # dict[str, Tensor] saved as .pt/.json appropriately
                        "timing": {
                            "score_aux_datasets_seconds": elapsed_time,
                        },
                    }

                # Use provided run_info
                info = run_info

                run_dir = result_store.write_run(
                    method_name=method_name,
                    target_task=tgt_name,
                    scores={k: float(v) for k, v in scores.items()},
                    artifacts=artifacts,
                    run_info=info,
                )
                print(f"[Saved attribution results to {run_dir}]")
            else:
                print(f"[one-step rank {rank}] Skipping save (only rank 0 saves)")

        return scores