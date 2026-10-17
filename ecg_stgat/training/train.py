"""
Training Engine for ST-GAT and Baseline Models on PTB-XL.
Implements BCEWithLogitsLoss with positive class weighting, AdamW, and early stopping.
"""
from typing import Dict, Any, Optional
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from ecg_stgat.data.ptbxl_dataset import get_ptbxl_dataloaders, DIAGNOSTIC_CLASSES
from ecg_stgat.models.stgat_model import STGATModel
from ecg_stgat.models.baseline_cnn import BaselineCNNModel


def build_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    m_cfg = cfg.get("model", {})
    name = m_cfg.get("name", "stgat").lower()
    embed_dim = m_cfg.get("embed_dim", 128)
    num_classes = m_cfg.get("num_classes", 5)
    dropout = m_cfg.get("dropout", 0.2)

    if name == "baseline_cnn":
        return BaselineCNNModel(embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)
    elif name == "stgat":
        return STGATModel(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=m_cfg.get("num_heads", 4),
            num_gat_layers=m_cfg.get("num_gat_layers", 2),
            graph_type=m_cfg.get("graph_type", "static"),
            fusion_type=m_cfg.get("fusion_type", "gated"),
            pooling_type=m_cfg.get("pooling_type", "attention"),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def train_experiment(
    config: Dict[str, Any],
    exp_name: str = "experiment",
    output_base_dir: str = "runs",
) -> Dict[str, Any]:
    """
    Runs training and validation for the specified configuration.
    Saves best model and logs under output_base_dir / exp_name.
    """
    exp_dir = Path(output_base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    device_str = config.get("training", {}).get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    data_cfg = config.get("data", {})
    batch_size = data_cfg.get("batch_size", 64)
    sampling_rate = data_cfg.get("sampling_rate", 100)
    train_folds = data_cfg.get("train_folds", list(range(1, 9)))
    val_folds = data_cfg.get("val_folds", [9])
    test_folds = data_cfg.get("test_folds", [10])
    data_dir = data_cfg.get("data_dir", "data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")

    print(f"[{exp_name}] Initializing DataLoaders (Train Folds: {train_folds}, Val: {val_folds})...")
    train_l, val_l, test_l, pos_w, _ = get_ptbxl_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        sampling_rate=sampling_rate,
        train_folds=train_folds,
        val_folds=val_folds,
        test_folds=test_folds,
        preprocess=True,
    )

    model = build_model_from_config(config).to(device)
    pos_w = pos_w.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = int(train_cfg.get("epochs", 10))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    best_epoch = 0
    history = []

    print(f"[{exp_name}] Starting Training for {epochs} epochs on device: {device}...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for x_b, y_b, _ in train_l:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / max(num_train_batches, 1)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_b, y_b, _ in val_l:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                logits = model(x_b)
                v_loss = criterion(logits, y_b)
                total_val_loss += v_loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(y_b.cpu().numpy())

        avg_val_loss = total_val_loss / max(len(val_l), 1)
        val_preds_arr = np.concatenate(val_preds, axis=0)
        val_targets_arr = np.concatenate(val_targets, axis=0)

        try:
            val_macro_auc = roc_auc_score(val_targets_arr, val_preds_arr, average="macro")
        except Exception:
            val_macro_auc = 0.5

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro AUROC: {val_macro_auc:.4f} | Time: {elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_macro_auc": val_macro_auc,
        })

        if val_macro_auc > best_val_auc or epoch == 1:
            best_val_auc = val_macro_auc
            best_epoch = epoch
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "best_val_auc": best_val_auc,
                "classes": DIAGNOSTIC_CLASSES,
            }, exp_dir / "best_model.pt")

    # Save history and metrics
    history_df = pd.DataFrame(history)
    history_df.to_csv(exp_dir / "training_history.csv", index=False)

    metrics = {
        "exp_name": exp_name,
        "best_epoch": best_epoch,
        "best_val_macro_auc": float(best_val_auc),
        "final_train_loss": float(avg_train_loss),
        "final_val_loss": float(avg_val_loss),
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{exp_name}] Completed! Best Model Checkpoint saved to: {exp_dir / 'best_model.pt'}\n")
    return metrics
