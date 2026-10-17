"""
Training ST-GAT with Stochastic Lead-Dropout Regularization (RQ5 / H4).
Randomly drops leads during training to build resilience against missing channels.
"""
from typing import Dict, Any
from pathlib import Path
import random
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from ecg_stgat.data.ptbxl_dataset import get_ptbxl_dataloaders, DIAGNOSTIC_CLASSES
from ecg_stgat.training.train import build_model_from_config


def apply_stochastic_lead_dropout(
    x: torch.Tensor,
    lead_mask_prob: float = 0.5,
    max_dropped_leads: int = 6,
) -> torch.Tensor:
    """
    Randomly drops up to max_dropped_leads with probability lead_mask_prob per sample.
    """
    if lead_mask_prob <= 0:
        return x

    B, num_leads, T = x.shape
    masked_x = x.clone()

    for b in range(B):
        if random.random() < lead_mask_prob:
            num_drop = random.randint(1, max_dropped_leads)
            drop_indices = random.sample(range(num_leads), num_drop)
            masked_x[b, drop_indices, :] = 0.0

    return masked_x


def train_lead_dropout_model(
    config_path: str = "ecg_stgat/training/config.yaml",
    lead_mask_prob: float = 0.5,
    max_dropped_leads: int = 6,
    exp_name: str = "STGAT_LeadDropout",
    epochs: int = 25,
    output_base_dir: str = "runs",
) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["training"]["epochs"] = epochs
    cfg["model"]["name"] = "stgat"
    cfg["model"]["graph_type"] = "static"

    exp_dir = Path(output_base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    data_cfg = cfg.get("data", {})
    train_l, val_l, _, pos_w, _ = get_ptbxl_dataloaders(
        data_dir=data_cfg.get("data_dir", "data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"),
        batch_size=data_cfg.get("batch_size", 64),
        train_folds=data_cfg.get("train_folds", list(range(1, 9))),
        val_folds=data_cfg.get("val_folds", [9]),
        preprocess=True,
    )

    model = build_model_from_config(cfg).to(device)
    pos_w = pos_w.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_auc = -1.0
    best_epoch = 0
    history = []

    print(f"[{exp_name}] Training with Lead-Dropout (p={lead_mask_prob}, max_drop={max_dropped_leads})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for x_b, y_b, _ in train_l:
            x_b_dropped = apply_stochastic_lead_dropout(x_b, lead_mask_prob, max_dropped_leads)
            x_b_dropped = x_b_dropped.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(x_b_dropped)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / max(num_batches, 1)

        # Validation (unmasked)
        model.eval()
        total_val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b, _ in val_l:
                logits = model(x_b.to(device))
                loss = criterion(logits, y_b.to(device))
                total_val_loss += loss.item()
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(y_b.numpy())

        avg_val_loss = total_val_loss / max(len(val_l), 1)
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_auc = roc_auc_score(val_targets, val_preds, average="macro")

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Val AUROC: {val_auc:.4f}")
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_macro_auc": val_auc})

        if val_auc > best_val_auc or epoch == 1:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "best_val_auc": best_val_auc,
                "classes": DIAGNOSTIC_CLASSES,
            }, exp_dir / "best_model.pt")

    pd.DataFrame(history).to_csv(exp_dir / "training_history.csv", index=False)

    metrics = {
        "exp_name": exp_name,
        "best_epoch": best_epoch,
        "best_val_macro_auc": float(best_val_auc),
        "final_train_loss": float(avg_train_loss),
        "final_val_loss": float(avg_val_loss),
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Evaluate on Test Fold 10 with bootstrap
    print(f"\n[{exp_name}] Evaluating checkpoint on Test Fold 10...")
    from ecg_stgat.evaluation.evaluate import evaluate_checkpoint
    evaluate_checkpoint(str(exp_dir / "best_model.pt"), bootstrap=True, bootstrap_iterations=500)

    return metrics
