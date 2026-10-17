"""
Test Fold 10 Evaluation Engine with Bootstrap Confidence Intervals.
"""
from typing import Dict, Any, Optional
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from ecg_stgat.data.ptbxl_dataset import get_ptbxl_dataloaders, DIAGNOSTIC_CLASSES
from ecg_stgat.training.train import build_model_from_config
from ecg_stgat.evaluation.metrics import compute_metrics, compute_bootstrap_ci


def evaluate_checkpoint(
    checkpoint_path: str,
    bootstrap: bool = True,
    bootstrap_iterations: int = 1000,
    data_dir: str = "data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Evaluates a trained checkpoint on official Test Fold 10.
    """
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    checkpoint = torch.load(ckpt_p, map_location=device)
    config = checkpoint.get("config", {
        "model": {"name": "stgat" if "stgat" in str(ckpt_p).lower() else "baseline_cnn"}
    })

    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\n[Evaluate] Running inference on Test Fold 10 for: {ckpt_p.name}...")
    _, _, test_loader, _, _ = get_ptbxl_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        test_folds=[10],
        preprocess=True,
    )

    y_true_list = []
    y_prob_list = []

    with torch.no_grad():
        for x_b, y_b, _ in test_loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob_list.append(probs)
            y_true_list.append(y_b.numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)

    metrics = compute_metrics(y_true, y_prob)

    if bootstrap:
        print(f"[Evaluate] Computing {bootstrap_iterations} Bootstrap Confidence Intervals...")
        ci = compute_bootstrap_ci(y_true, y_prob, n_iterations=min(bootstrap_iterations, 500))
        metrics.update(ci)

    # Save metrics in checkpoint directory
    out_file = ckpt_p.parent / "test_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Evaluate] Results for {ckpt_p.parent.name}:")
    print(f"  Macro AUROC: {metrics['macro_auc']:.4f}")
    print(f"  Micro AUROC: {metrics['micro_auc']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Micro F1:    {metrics['micro_f1']:.4f}")
    print(f"  Saved to:    {out_file}\n")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint)
