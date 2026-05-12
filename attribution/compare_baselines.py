"""
Compare baseline methods across multiple k values using Schulze voting.

This script automatically extracts metrics from final_metrics.json files using
the metric infrastructure from attribution/metrics/. 

Key improvements:
- Automatically infers metrics from metric_info in final_metrics.json
- Uses orientation.py to determine if higher/lower is better
- Supports both HF trainer metrics and lm-eval metrics via LMEvalMetricSpec
- Handles multiple metrics simultaneously (not just accuracy/loss)
- Works with any metrics that have orientations defined

Usage:
    python compare_baselines.py --root <path_to_methods> [--output rankings.json]

Expected directory structure:
    <root>/
        method1/
            train_k10/
                final_metrics.json
            train_k20/
                final_metrics.json
        method2/
            train_k10/
                final_metrics.json
            train_k20/
                final_metrics.json

The final_metrics.json should contain:
    {
        "all_eval_results": {
            "hf": {...},      # Optional: HF trainer metrics
            "lmeval": {...}   # Optional: lm-eval results
        },
        "metric_info": {
            "type": "lm_eval" | "hf_trainer",
            "metric_specs": [...],         # For lm_eval
            "available_metrics": [...],    # For hf_trainer
            "orientations": {...}          # For hf_trainer
        }
    }
"""
import os, re, json, argparse
from dataclasses import dataclass
import math
from typing import Dict, List, Tuple, Optional, Set
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics.orientation import get_metric_orientation

@dataclass
class Metrics:
    """Store all metrics from a run with their orientations."""
    metrics: Dict[str, float]  # metric_name -> value
    orientations: Dict[str, int]  # metric_name -> +1/-1

def read_metrics(path: str) -> Optional[Metrics]:
    """
    Read ALL metrics from final_metrics.json from both HF and lm-eval sources.
    
    This extracts metrics comprehensively regardless of metric_info.type:
    - HF trainer metrics from all_eval_results.hf
    - lm-eval metrics from all_eval_results.lmeval
    - Uses metric_info when available for orientation hints
    
    Returns:
        Metrics object with all available metrics and their orientations
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)

        all_eval = data.get("all_eval_results", {})
        if not all_eval:
            return None
        
        metrics_dict = {}
        orientations_dict = {}
        
        # Get metric_info for hints (orientations, metric_specs)
        metric_info = data.get("metric_info", {})
        metric_specs_data = metric_info.get("metric_specs", [])
        
        # ============ Extract HF metrics ============
        hf = all_eval.get("hf", {})
        if isinstance(hf, dict):
            for metric_name, value in hf.items():
                if isinstance(value, (int, float)) and not math.isinf(value):
                    # Skip metadata metrics that shouldn't be ranked
                    if metric_name in ["epoch", "step", "num_tokens", "num_samples"]:
                        continue
                    metrics_dict[metric_name] = float(value)
                    # Always infer orientation
                    orientations_dict[metric_name] = get_metric_orientation(metric_name)
        
        # ============ Extract lm-eval metrics ============
        lmeval = all_eval.get("lmeval", {})
        if isinstance(lmeval, dict):
            results = lmeval.get("results", {})
            if isinstance(results, dict):
                # If we have metric_specs, use them for orientation info
                specs_map = {}  # (dataset, metric_name) -> higher_is_better
                if metric_specs_data:
                    for dataset, metric_name, higher_is_better in metric_specs_data:
                        specs_map[(dataset, metric_name)] = higher_is_better
                
                # Extract all lm-eval metrics
                for dataset, dataset_metrics in results.items():
                    if isinstance(dataset_metrics, dict):
                        for metric_name, value in dataset_metrics.items():
                            if isinstance(value, (int, float)):
                                # Skip stderr metrics (uncertainty measures, not performance)
                                if "_stderr" in metric_name:
                                    continue
                                
                                # Create unique key: dataset:metric_name
                                key = f"{dataset}:{metric_name}"
                                metrics_dict[key] = float(value)
                                
                                # Determine orientation
                                if (dataset, metric_name) in specs_map:
                                    # Use orientation from metric_specs
                                    orientations_dict[key] = 1 if specs_map[(dataset, metric_name)] else -1
                                else:
                                    # Infer from metric name
                                    orientations_dict[key] = get_metric_orientation(metric_name)
        
        # ============ Extract default metrics (fallback) ============
        if not metrics_dict:
            default = all_eval.get("default", {})
            if isinstance(default, dict):
                for metric_name, value in default.items():
                    if isinstance(value, (int, float)) and not math.isinf(value):
                        metrics_dict[metric_name] = float(value)
                        orientations_dict[metric_name] = get_metric_orientation(metric_name)
        
        if not metrics_dict:
            return None
        
        return Metrics(metrics=metrics_dict, orientations=orientations_dict)

    except Exception as e:
        print(f"Warning: Failed to read metrics from {path}: {e}")
        return None


def find_k_from_dirname(dirname: str) -> Optional[int]:
    m = re.search(r"train_k(\d+)", dirname)
    return int(m.group(1)) if m else None


def gather_all(
    root: str,
    metrics_file: str,
) -> Tuple[Dict[int, Dict[str, Metrics]], Dict[str, Set[int]], Set[str]]:
    """
    Gather all metrics from all methods and runs.
    
    Returns:
        per_k_raw[k][method] = Metrics
        method_to_ks[method] = set of ks available for that method  
        all_metric_names = set of all metric names seen across all runs
    """
    per_k_raw: Dict[int, Dict[str, Metrics]] = {}
    method_to_ks: Dict[str, Set[int]] = {}
    all_metric_names: Set[str] = set()

    for method_entry in os.scandir(root):
        if not method_entry.is_dir():
            continue
        method = method_entry.name
        ks_found: Set[int] = set()

        for run_entry in os.scandir(method_entry.path):
            if not run_entry.is_dir():
                continue
            k = find_k_from_dirname(run_entry.name)
            if k is None:
                continue

            metrics_path = os.path.join(run_entry.path, metrics_file)
            if not os.path.isfile(metrics_path):
                continue

            m = read_metrics(metrics_path)
            if m is None:
                continue

            per_k_raw.setdefault(k, {})[method] = m
            ks_found.add(k)
            all_metric_names.update(m.metrics.keys())

        if ks_found:
            method_to_ks[method] = ks_found

    return per_k_raw, method_to_ks, all_metric_names


def filter_full_coverage_union(
    per_k_raw: Dict[int, Dict[str, Metrics]],
    method_to_ks: Dict[str, Set[int]],
):
    """
    Use the UNION of all ks observed across methods as the required set.
    Keep ONLY methods that have results for ALL ks in this union.
    Return filtered per_k, kept methods list, and sorted ks_required.
    """
    if not method_to_ks:
        return {}, [], []

    ks_required: Set[int] = set()
    for ks in method_to_ks.values():
        ks_required |= ks
    if not ks_required:
        return {}, [], []

    methods_full = sorted([m for m, ks in method_to_ks.items() if ks_required.issubset(ks)])
    if not methods_full:
        return {}, [], []

    per_k: Dict[int, Dict[str, Metrics]] = {}
    for k in ks_required:
        if k not in per_k_raw:
            continue
        row = {m: per_k_raw[k][m] for m in methods_full if m in per_k_raw[k]}
        if len(row) == len(methods_full):
            per_k[k] = row

    ks_final = sorted(per_k.keys())
    if not ks_final:
        return {}, [], []
    return per_k, methods_full, ks_final


def rank_by_metric(
    method_metrics: Dict[str, Metrics],
    metric_name: str,
) -> List[str]:
    """
    Rank methods by a specific metric, respecting its orientation.
    
    For higher-is-better metrics: higher values ranked first
    For lower-is-better metrics: lower values ranked first
    Ties broken by method name for stability
    """
    def sort_key(method: str):
        metrics = method_metrics[method]
        value = metrics.metrics.get(metric_name)
        if value is None:
            # Missing values go to the end
            return (float('inf'), method) if metrics.orientations.get(metric_name, 1) > 0 else (float('-inf'), method)
        
        orientation = metrics.orientations.get(metric_name, 1)
        # Negate if higher is better (to get descending order)
        return (-value if orientation > 0 else value, method)
    
    return sorted(method_metrics.keys(), key=sort_key)


def build_pairwise_from_rankings(rankings: List[List[str]], methods: List[str]) -> Dict[Tuple[str, str], int]:
    """
    rankings: list of complete, tie-free rankings (best->worst).
    Returns d[(A,B)] = #rankings that prefer A over B.
    """
    d = {(a, b): 0 for a in methods for b in methods if a != b}
    for r in rankings:
        pos = {m: i for i, m in enumerate(r)}
        for a in methods:
            for b in methods:
                if a == b:
                    continue
                if pos[a] < pos[b]:
                    d[(a, b)] += 1
    return d


def schulze_tiers(methods: List[str], d: Dict[Tuple[str, str], int]) -> List[List[str]]:
    """
    Return Schulze ranking as TIERS (list of lists), not a forced total order.
    Tier 0 is best. Methods within the same tier are tied under Schulze.
    """
    # ---- Standard Schulze strongest-path computation ----
    p: Dict[Tuple[str, str], int] = {}
    for a in methods:
        for b in methods:
            if a == b:
                continue
            dab = d.get((a, b), 0)
            dba = d.get((b, a), 0)
            p[(a, b)] = dab if dab > dba else 0

    for i in methods:
        for j in methods:
            if i == j:
                continue
            for k in methods:
                if i == k or j == k:
                    continue
                p[(j, k)] = max(p[(j, k)], min(p[(j, i)], p[(i, k)]))

    # ---- Build strict-preference graph: a -> b if a beats b under Schulze ----
    remaining = set(methods)
    tiers: List[List[str]] = []

    # Precompute who beats whom for speed/clarity
    beats = {a: set() for a in methods}
    for a in methods:
        for b in methods:
            if a == b:
                continue
            if p[(a, b)] > p[(b, a)]:
                beats[a].add(b)

    while remaining:
        # In-degree within remaining set
        indeg = {a: 0 for a in remaining}
        for a in remaining:
            for b in beats[a]:
                if b in remaining:
                    indeg[b] += 1

        # Current best tier = nobody beats them (in-degree 0)
        tier = sorted([a for a, deg in indeg.items() if deg == 0])

        # Safety fallback (shouldn't happen if relation is a proper partial order)
        if not tier:
            tier = sorted(list(remaining))

        tiers.append(tier)
        remaining -= set(tier)

    return tiers


def best_over_k(
    per_k: Dict[int, Dict[str, Metrics]],
    methods: List[str],
    ks: List[int],
    metric_names: Set[str],
):
    """
    For each method and metric, find the k value that gives the best performance.
    Also compute the overall best method for each metric.
    
    Returns:
      {
        "per_method": {
          method: {
            metric_name: {"k": k, "value": val} or None if not available
          }
        },
        "per_metric": {
          metric_name: {
            "best_method": method,
            "k": k,
            "value": val
          }
        }
      }
    """
    per_method = {}
    per_metric = {}

    for m in methods:
        method_best = {}
        
        for metric_name in sorted(metric_names):
            best_k = None
            best_value = None
            
            # Determine if we should maximize or minimize
            # Get orientation from first available metrics object
            orientation = None
            for k in ks:
                if k in per_k and m in per_k[k]:
                    orientation = per_k[k][m].orientations.get(metric_name)
                    if orientation is not None:
                        break
            
            if orientation is None:
                # Metric not available for this method
                method_best[metric_name] = None
                continue
            
            maximize = orientation > 0
            
            for k in ks:
                if k not in per_k or m not in per_k[k]:
                    continue
                
                met = per_k[k][m]
                value = met.metrics.get(metric_name)
                
                if value is None or (isinstance(value, float) and math.isinf(value)):
                    continue
                
                if best_k is None:
                    best_k = k
                    best_value = value
                else:
                    if maximize and value > best_value:
                        best_k = k
                        best_value = value
                    elif not maximize and value < best_value:
                        best_k = k
                        best_value = value
            
            if best_k is not None:
                method_best[metric_name] = {
                    "k": int(best_k),
                    "value": float(best_value),
                }
            else:
                method_best[metric_name] = None
        
        per_method[m] = method_best
    
    # Now find the best method for each metric
    for metric_name in sorted(metric_names):
        best_method = None
        best_k = None
        best_value = None
        orientation = None
        
        # Find orientation from any method
        for m in methods:
            if metric_name in per_method[m] and per_method[m][metric_name] is not None:
                for k in ks:
                    if k in per_k and m in per_k[k]:
                        orientation = per_k[k][m].orientations.get(metric_name)
                        if orientation is not None:
                            break
                if orientation is not None:
                    break
        
        if orientation is None:
            continue
        
        maximize = orientation > 0
        
        # Find best across all methods
        for m in methods:
            if metric_name not in per_method[m] or per_method[m][metric_name] is None:
                continue
            
            m_best = per_method[m][metric_name]
            m_value = m_best["value"]
            m_k = m_best["k"]
            
            if best_method is None:
                best_method = m
                best_value = m_value
                best_k = m_k
            else:
                if maximize and m_value > best_value:
                    best_method = m
                    best_value = m_value
                    best_k = m_k
                elif not maximize and m_value < best_value:
                    best_method = m
                    best_value = m_value
                    best_k = m_k
        
        if best_method is not None:
            per_metric[metric_name] = {
                "best_method": best_method,
                "k": int(best_k),
                "value": float(best_value),
            }

    return {"per_method": per_method, "per_metric": per_metric}

def format_tiers(tiers: List[List[str]]) -> str:
    return " ≻ ".join(["{" + ", ".join(t) + "}" for t in tiers])

def main():
    ap = argparse.ArgumentParser(
        description="Per-k rankings for all metrics + Schulze aggregation into one JSON."
    )
    ap.add_argument("--root", type=str, required=True, help="Root dir with <method>/train_k*/final_metrics.json")
    ap.add_argument("--metrics-file", type=str, default="final_metrics.json", help="Metrics filename inside each train_k*/")
    ap.add_argument("--output", type=str, default="rankings_schulze.json", help="Single JSON output path")

    args = ap.parse_args()

    per_k_raw, method_to_ks, all_metric_names = gather_all(
        args.root,
        args.metrics_file,
    )
    if not per_k_raw:
        print("No metrics found. Check --root and --metrics-file.")
        return

    per_k, methods, ks = filter_full_coverage_union(per_k_raw, method_to_ks)
    if not per_k:
        print("After filtering for FULL k coverage (union), nothing remains.")
        print("Tip: ensure all compared methods have the same set of train_k* folders,")
        print("and that each final_metrics.json contains valid metrics.")
        return
    
    # Find common metrics across all methods at all k values
    common_metrics = set(all_metric_names)
    for k in ks:
        for method in methods:
            if k in per_k and method in per_k[k]:
                common_metrics &= set(per_k[k][method].metrics.keys())
    
    if not common_metrics:
        print("No common metrics found across all methods and k values.")
        print("Available metrics vary across runs. Proceeding with all metrics (rankings may be incomplete).")
        common_metrics = all_metric_names

    print(f"Found {len(common_metrics)} metric(s) to rank: {sorted(common_metrics)}")

    # Build per-k rankings for each metric
    per_k_rankings = {}
    metric_ballots: Dict[str, List[List[str]]] = {m: [] for m in common_metrics}

    for k in ks:
        k_rankings = {}
        for metric_name in sorted(common_metrics):
            ranking = rank_by_metric(per_k[k], metric_name)
            k_rankings[metric_name] = ranking
            metric_ballots[metric_name].append(ranking)
        
        per_k_rankings[int(k)] = k_rankings

    # Compute Schulze order for each metric
    schulze_orders = {}
    for metric_name in sorted(common_metrics):
        d = build_pairwise_from_rankings(metric_ballots[metric_name], methods)
        tiers = schulze_tiers(methods, d)
        schulze_orders[metric_name] = tiers  # store tiers directly

    # Compute best k for each method and metric
    best_summary = best_over_k(per_k, methods, ks, common_metrics)

    out_obj = {
        "methods": methods,
        "ks": ks,
        "metrics": sorted(common_metrics),
        "per_k_rankings": per_k_rankings,
        "schulze": schulze_orders,
        "best_summary": best_summary
    }
    out_path = os.path.join(args.root, args.output)
    with open(out_path, "w") as f:
        json.dump(out_obj, f, indent=2)

    print("Kept methods (full k coverage):", methods)
    print("Ks used:", ks)
    print("\nPer-k rankings:")
    for k in ks:
        print(f"  k={k}:")
        for metric_name in sorted(common_metrics):
            print(f"    {metric_name:20s}: " + " > ".join(per_k_rankings[int(k)][metric_name]))
    print("\nSchulze aggregate orders across k:")
    for metric_name in sorted(common_metrics):
        print(f"  {metric_name:20s}: " + format_tiers(schulze_orders[metric_name]))
    
    print("\n" + "="*80)
    print("BEST METHOD FOR EACH METRIC (across all k values)")
    print("="*80)
    for metric_name in sorted(common_metrics):
        if metric_name in best_summary["per_metric"]:
            best_info = best_summary["per_metric"][metric_name]
            print(f"{metric_name:30s}: {best_info['best_method']:20s} (k={best_info['k']}, value={best_info['value']:.4f})")
        else:
            print(f"{metric_name:30s}: No data available")
    
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()