"""
Missing-Lead Robustness Benchmark (RQ4).
Evaluates performance degradation when dropping leads down to 12, 10, 8, 6, 3, and 1 lead.
"""
from typing import Dict, List, Any
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from ecg_stgat.data.ptbxl_dataset import get_ptbxl_dataloaders
from ecg_stgat.training.train import build_model_from_config
from ecg_stgat.evaluation.metrics import compute_metrics


LEAD_SUBSETS = {
    "12_leads": list(range(12)),
    "10_leads": [0, 1, 3, 4, 6, 7, 8, 9, 10, 11],  # Drop III, aVF
    "8_leads": [0, 1, 3, 4, 6, 8, 10, 11],
    "6_leads": [0, 1, 2, 3, 4, 5],                 # Standard Limb Leads
    "3_leads": [0, 1, 2],                          # Einthoven Triangle I, II, III
    "1_lead": [1],                                 # Single Lead II
}


def run_missing_lead_benchmark(
    checkpoint_path: str,
    output_dir: str = "runs/missing_leads",
    data_dir: str = "data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Evaluates model across reduced lead subsets on Test Fold 10.
    """
    ckpt_p = Path(checkpoint_path)
    model_name = ckpt_p.parent.name
    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load(ckpt_p, map_location=device)
    config = checkpoint.get("config", {
        "model": {"name": "stgat" if "stgat" in str(ckpt_p).lower() else "baseline_cnn"}
    })

    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader, _, _ = get_ptbxl_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        test_folds=[10],
        preprocess=True,
    )

    results = []
    print(f"\n[Missing-Lead Benchmark] Evaluating {model_name}...")

    for config_name, active_leads in LEAD_SUBSETS.items():
        y_true_list = []
        y_prob_list = []

        with torch.no_grad():
            for x_b, y_b, _ in test_loader:
                # Mask out inactive leads with zeros
                mask = torch.zeros_like(x_b)
                mask[:, active_leads, :] = 1.0
                masked_x = x_b * mask

                logits = model(masked_x.to(device))
                probs = torch.sigmoid(logits).cpu().numpy()
                y_prob_list.append(probs)
                y_true_list.append(y_b.numpy())

        y_true = np.concatenate(y_true_list, axis=0)
        y_prob = np.concatenate(y_prob_list, axis=0)

        m = compute_metrics(y_true, y_prob)
        num_active = len(active_leads)
        results.append({
            "model": model_name,
            "configuration": config_name,
            "num_leads": num_active,
            "macro_auc": m["macro_auc"],
            "micro_auc": m["micro_auc"],
            "macro_f1": m["macro_f1"],
            "micro_f1": m["micro_f1"],
            "exact_match_accuracy": m["exact_match_accuracy"],
        })
        print(f"  {config_name:10s} ({num_active:2d} leads) -> Macro AUROC: {m['macro_auc']:.4f} | Macro F1: {m['macro_f1']:.4f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "missing_lead_results.csv", index=False)
    return df_res
