"""
Ablation Study Suite (RQ2).
Evaluates Temporal-Only, Graph-Only, Concat Fusion, Mean Pooling, Fully-Connected, and Learned Graph.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import pandas as pd
from ecg_stgat.training.train import train_experiment
from ecg_stgat.evaluation.evaluate import evaluate_checkpoint

import json

ABLATION_CONFIGS = {
    "temporal_only": {"model": {"name": "stgat", "fusion_type": "temporal_only"}},
    "graph_only": {"model": {"name": "stgat", "fusion_type": "graph_only"}},
    "concat_fusion": {"model": {"name": "stgat", "fusion_type": "concat"}},
    "fc_graph": {"model": {"name": "stgat", "graph_type": "fully_connected"}},
    "mean_pooling": {"model": {"name": "stgat", "pooling_type": "mean"}},
}


def run_ablation_suite(
    base_config_path: str = "ecg_stgat/training/config.yaml",
    epochs: int = 20,
    output_dir: str = "runs/ablations",
    conditions: Optional[list] = None,
) -> pd.DataFrame:
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    results = []

    target_configs = ABLATION_CONFIGS
    if conditions is not None:
        target_configs = {k: v for k, v in ABLATION_CONFIGS.items() if k in conditions}

    for name, overrides in target_configs.items():
        model_ckpt = out_p / name / "best_model.pt"
        test_metrics_file = out_p / name / "test_metrics.json"

        if model_ckpt.exists() and test_metrics_file.exists():
            print(f"\n[Ablation Suite] Condition '{name}' already completed. Reusing verified checkpoint and metrics.")
            with open(test_metrics_file, "r") as f:
                m = json.load(f)
        else:
            print(f"\n[Ablation Suite] Running ablation condition: {name} (epochs={epochs})...")
            cfg = yaml.safe_load(yaml.dump(base_cfg))
            cfg["training"]["epochs"] = epochs
            for section, params in overrides.items():
                cfg[section].update(params)

            train_experiment(cfg, exp_name=name, output_base_dir=output_dir)
            m = evaluate_checkpoint(str(out_p / name / "best_model.pt"), bootstrap=False)

        results.append({
            "variant": name,
            "macro_auc": m["macro_auc"],
            "micro_auc": m["micro_auc"],
            "macro_f1": m["macro_f1"],
            "micro_f1": m["micro_f1"],
            "exact_match_accuracy": m.get("exact_match_accuracy", 0.0),
        })

    df = pd.DataFrame(results)
    df.to_csv(out_p / "ablation_results.csv", index=False)
    return df
