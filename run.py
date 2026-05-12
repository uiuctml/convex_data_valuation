import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_utils import load_yaml_config
from trainer.trainer_factory import make_trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_yaml_config(args.cfg)
    trainer = make_trainer(cfg=cfg)

    try:
        trainer.train()
        print("[EVAL] Starting evaluation...")
        final_metrics = trainer.evaluate()
        
        if trainer.is_world_process_zero():
            print("[FINAL] ", final_metrics)
            metrics_path = os.path.join(cfg.output_dir, "final_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(final_metrics, f, indent=2)
            print(f"[save] Final metrics saved to {metrics_path}")
        if not getattr(cfg, 'eval_with_lmeval', False):
            trainer.save_model()
    
    finally:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
            print("[cleanup] Destroyed distributed process group")

if __name__ == "__main__":
    main()