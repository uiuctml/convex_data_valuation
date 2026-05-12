from __future__ import annotations
from typing import List, Optional, Dict, Sequence
import os
import numpy as np
import torch
import cvxpy as cp

from attribution.gradient.base import GradientAttributionBase
from attribution.save import ResultStore, AttributionRunInfo


class GradientKMMAttribution(GradientAttributionBase):
    """
    Kernel Mean Matching in gradient/NTK space with L1 (LASSO), solved via CVXPY.

    Loads precomputed flattened gradients from an artifacts directory.

    Supported layouts:
    - one_step / artifacts:
        target_avg_grad.pt              # Tensor (d,)
        aux_avg_grads.pt                # Dict[str, Tensor(d,)]
    - task_vector:
        target_task_vector.pt           # Tensor (d,)
        aux_<task_name>_task_vector.pt  # Tensor (d,) per aux

    Solves (Gram form):  min_beta  0.5*beta^T K beta - c^T beta + lam * ||beta||_1
      where K = A^T A (n x n), c = A^T a (n,).
    """

    def __init__(
        self,
        trainer_factory,
        # ---- where to load gradients ----
        artifact_dir: str = "",
        source_preference: Sequence[str] = ("one_step", "task_vector"),
        # ---- optimization knobs ----
        l1_lambda: float = 1e-2,
        elastic_net_gamma: float = 0.0,   # small ridge on beta if K is ill-conditioned (e.g., 1e-6)
        normalize_vectors: bool = False,  # if True, L2-normalize each vector before K,c construction
        solver: str = "OSQP",             # OSQP, ECOS, or SCS
        solver_kwargs: Optional[Dict] = None,
        verbose: bool = True,
    ):
        super().__init__(trainer_factory=trainer_factory)
        self.artifact_dir = artifact_dir
        self.source_preference = list(source_preference)
        self.l1_lambda = float(l1_lambda)
        self.elastic_net_gamma = float(elastic_net_gamma)
        self.normalize_vectors = bool(normalize_vectors)
        self.solver = solver
        self.solver_kwargs = solver_kwargs or {}
        self.verbose = verbose

    # ---------- loading helpers ----------
    def _state_dict_to_flat(self, state_dict: dict) -> torch.Tensor:
        """Convert a gradient state_dict to a flattened tensor."""
        return torch.cat([state_dict[k].reshape(-1) for k in sorted(state_dict.keys())], dim=0)

    def _load_one_step(self, aux_names: List[str], target_task_name: Optional[str]):
        t_path = os.path.join(self.artifact_dir, "target_avg_grad.pt")
        a_path = os.path.join(self.artifact_dir, "aux_avg_grads.pt")
        if not (os.path.isfile(t_path) and os.path.isfile(a_path)):
            return None, None
        tgt = torch.load(t_path, map_location="cpu")
        aux_dict = torch.load(a_path, map_location="cpu")
        
        # Convert state_dict to flat tensor if needed
        if isinstance(tgt, dict):
            tgt = self._state_dict_to_flat(tgt)
        
        missing = [n for n in aux_names if n not in aux_dict]
        if missing:
            print("[KMM/L1][one_step] Missing aux in aux_avg_grads.pt:", ", ".join(missing))
            return None, None
        
        aux_vecs = []
        for n in aux_names:
            aux_grad = aux_dict[n]
            # Convert state_dict to flat tensor if needed
            if isinstance(aux_grad, dict):
                aux_grad = self._state_dict_to_flat(aux_grad)
            aux_vecs.append(aux_grad)
        
        return tgt, aux_vecs

    def _load_task_vector(self, aux_names: List[str], target_task_name: Optional[str]):
        t_path = os.path.join(self.artifact_dir, "target_task_vector.pt")
        if not os.path.isfile(t_path):
            return None, None
        tgt = torch.load(t_path, map_location="cpu")
        
        # Convert state_dict to flat tensor if needed
        if isinstance(tgt, dict):
            tgt = self._state_dict_to_flat(tgt)
        
        aux_vecs, missing = [], []
        for n in aux_names:
            pattern = os.path.join(self.artifact_dir, f"aux_{n}_task_vector.pt")
            if not os.path.isfile(pattern):
                missing.append(n)
            else:
                aux_tv = torch.load(pattern, map_location="cpu")
                # Convert state_dict to flat tensor if needed
                if isinstance(aux_tv, dict):
                    aux_tv = self._state_dict_to_flat(aux_tv)
                aux_vecs.append(aux_tv)
        if missing:
            print("[KMM/L1][task_vector] Missing aux task_vector files for:", ", ".join(missing))
            return None, None
        return tgt, aux_vecs

    def _load_precomputed(self, aux_names: List[str], target_task_name: Optional[str]):
        for src in self.source_preference:
            if src == "one_step":
                tgt, aux = self._load_one_step(aux_names, target_task_name)
            elif src == "task_vector":
                tgt, aux = self._load_task_vector(aux_names, target_task_name)
            else:
                continue
            if tgt is not None and aux is not None:
                if self.verbose:
                    print(f"[KMM/L1] Loaded precomputed gradients from '{src}' at {self.artifact_dir}")
                return tgt, aux, src
        return None, None, None

    def score_auxiliary_datasets(
        self,
        *,
        # ---- optional persistence hooks ----
        result_store: Optional[ResultStore] = None,
        run_info: Optional[AttributionRunInfo] = None,
        save_artifacts: bool = True,
    ) -> Dict[str, float]:
        """
        Returns aux_name -> beta_i (L1 KMM weights). Requires artifacts on disk.
        If artifacts are missing/incomplete, prints a message and returns {}.

        Uses Gram formulation (K=A^T A, c=A^T a) to avoid building A (d x n).
        """
        import time
        
        # Start timing for score_aux_datasets
        start_time = time.time()
        
        if not self.artifact_dir:
            print("[KMM/L1] artifact_dir not provided. Aborting.")
            return {}

        # Extract data from trainer_factory
        aux_loaders = self.factory.aux_loaders
        target_task_name = self.factory.target_task
        
        aux_names = list(aux_loaders.keys())
        if not aux_names:
            print("[KMM/L1] No auxiliary loaders provided. Aborting.")
            return {}

        tgt, aux_vecs, source_used = self._load_precomputed(aux_names, target_task_name)
        if tgt is None or aux_vecs is None:
            print("[KMM/L1] Could not load required precomputed gradients from artifact_dir. Aborting.")
            return {}

        # --- Prepare vectors (CPU float32), optional normalization ---
        def _prep(v: torch.Tensor) -> torch.Tensor:
            x = v.reshape(-1).detach().cpu().to(torch.float32)
            if self.normalize_vectors:
                nrm = torch.linalg.norm(x)
                if nrm > 0:
                    x = x / nrm
            return x

        a_vec = _prep(tgt)
        aux_list = [_prep(g) for g in aux_vecs]
        n = len(aux_list)

        # --- Build c = A^T a (n,) and K = A^T A (n x n) without stacking into A ---
        c = torch.empty(n, dtype=torch.float32)
        for i in range(n):
            c[i] = torch.dot(aux_list[i], a_vec)

        K = torch.empty((n, n), dtype=torch.float32)
        for i in range(n):
            K[i, i] = torch.dot(aux_list[i], aux_list[i])
            for j in range(i + 1, n):
                val = torch.dot(aux_list[i], aux_list[j])
                K[i, j] = val
                K[j, i] = val

        K_np = K.numpy()
        c_np = c.numpy()

        # Optional ridge (elastic net gamma * ||beta||_2^2)
        if self.elastic_net_gamma > 0.0:
            K_np = K_np + self.elastic_net_gamma * np.eye(n, dtype=K_np.dtype)

        # --- Solve with CVXPY: 0.5 β^T K β - c^T β + λ ||β||_1 ---
        lam = self.l1_lambda
        beta = cp.Variable(n)
        obj = 0.5 * cp.quad_form(beta, K_np) - c_np @ beta + lam * cp.norm1(beta)
        prob = cp.Problem(cp.Minimize(obj))

        try:
            prob.solve(solver=self.solver, **self.solver_kwargs)
        except Exception as e:
            print(f"[KMM/L1] CVXPY solve failed with solver={self.solver}: {e}")
            return {}

        if beta.value is None:
            print("[KMM/L1] CVXPY returned no solution (beta.value is None). Aborting.")
            return {}

        beta_np = np.asarray(beta.value).reshape(-1)
        scores: Dict[str, float] = {name: float(b) for name, b in zip(aux_names, beta_np.tolist())}

        # ---- Pretty print ----
        if self.verbose:
            # residual in Gram space: ||Aβ - a||^2 = β^T K β - 2 c^T β + a^T a
            a_norm2 = float((a_vec @ a_vec).item())
            fit = float(beta_np @ (K_np @ beta_np) - 2.0 * (c_np @ beta_np) + a_norm2)
            fit = max(fit, 0.0)
            resid = fit ** 0.5
            print(f"[KMM/L1] source='{source_used}'  λ={lam:.3g}  residual ||Aβ - a|| = {resid:.6e}")
            if self.elastic_net_gamma > 0:
                print(f"[KMM/L1] elastic_net_gamma={self.elastic_net_gamma:.2e}")
            if self.normalize_vectors:
                print("[KMM/L1] vectors normalized to unit L2 norm before building K,c")
            for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
                print(f"{k}: {v:.6f}")

        # ---- Optional persistence ----
        if result_store is not None:
            method_name = "kmm_l1"
            tgt_name = target_task_name or "target"
            
            # Record timing
            elapsed_time = time.time() - start_time

            artifacts = {}
            if save_artifacts:
                artifacts = {
                    "beta": torch.from_numpy(beta_np.astype(np.float32)),
                    "residual": resid if self.verbose else float("nan"),
                    "lambda": lam,
                    "source_used": source_used,
                    "elastic_net_gamma": self.elastic_net_gamma,
                    "normalized": self.normalize_vectors,
                    "timing": {
                        "score_aux_datasets_seconds": elapsed_time,
                    },
                }

            # Use provided run_info or raise error if missing
            if run_info is None:
                raise ValueError(
                    "run_info is required when result_store is provided. "
                    "Please pass an AttributionRunInfo instance."
                )

            run_dir = result_store.write_run(
                method_name=method_name,
                target_task=tgt_name,
                scores=scores,
                artifacts=artifacts,
                run_info=run_info,
            )
            if self.verbose:
                print(f"[Saved attribution results to {run_dir}]")

        return scores