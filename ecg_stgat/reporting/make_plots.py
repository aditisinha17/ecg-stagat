"""
Publication Figure Generator for ECG ST-GAT Research Paper.
Plots:
1. Training and Validation Loss / AUROC convergence curves
2. Per-class diagnostic comparison bar charts
3. Missing-lead degradation curves (12 -> 1 leads)
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_training_curves(
    stgat_history_csv: str = "runs/STGAT_Static/training_history.csv",
    baseline_history_csv: str = "runs/Baseline_CNN/training_history.csv",
    output_dir: str = "results/figures",
) -> None:
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    for path, label, color in [(stgat_history_csv, "ST-GAT (Static)", "#2563EB"), (baseline_history_csv, "Baseline 1D-CNN", "#DC2626")]:
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            axes[0].plot(df["epoch"], df["val_loss"], label=f"{label} Val Loss", color=color, linewidth=2)
            axes[1].plot(df["epoch"], df["val_macro_auc"], label=f"{label} Val AUROC", color=color, linewidth=2)

    axes[0].set_title("Validation Loss Convergence", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()

    axes[1].set_title("Validation Macro AUROC", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro AUROC")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_p / "fig1_convergence_curves.png", dpi=300)
    plt.close()


def plot_per_class_bar_chart(
    stgat_metrics_json: str = "runs/STGAT_Static/test_metrics.json",
    baseline_metrics_json: str = "runs/Baseline_CNN/test_metrics.json",
    output_dir: str = "results/figures",
) -> None:
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # Load actual evaluated per-class metrics
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    
    stgat_p = Path(stgat_metrics_json)
    if not stgat_p.exists():
        stgat_p = stgat_p.parent / "metrics.json"

    base_p = Path(baseline_metrics_json)
    if not base_p.exists():
        base_p = base_p.parent / "metrics.json"

    with open(stgat_p, "r") as f:
        stgat_data = json.load(f)
    with open(base_p, "r") as f:
        base_data = json.load(f)

    stgat_per_class = stgat_data.get("per_class_auc", {})
    base_per_class = base_data.get("per_class_auc", {})

    stgat_vals = [float(stgat_per_class.get(c, 0.5)) for c in classes]
    base_vals = [float(base_per_class.get(c, 0.5)) for c in classes]

    plt.figure(figsize=(9, 5), dpi=300)
    x = range(len(classes))
    width = 0.35
    bars_base = plt.bar([i - width/2 for i in x], base_vals, width=width, label="Baseline 1D-CNN", color="#DC2626", edgecolor="black", linewidth=0.6)
    bars_stgat = plt.bar([i + width/2 for i in x], stgat_vals, width=width, label="ST-GAT (Static Graph)", color="#2563EB", edgecolor="black", linewidth=0.6)

    # Annotate bars with real values
    for bar in bars_base:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., y + 0.008, f"{y:.3f}", ha="center", va="bottom", fontsize=8.5, rotation=0)
    for bar in bars_stgat:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., y + 0.008, f"{y:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", rotation=0)

    plt.xticks(x, classes, fontweight="bold")
    plt.ylabel("Test Set AUROC (Fold 10)", fontweight="bold")
    min_y = min(min(stgat_vals), min(base_vals))
    plt.ylim(max(0.0, min_y - 0.08), 1.05)
    plt.title("Per-Class AUROC Comparison: ST-GAT vs. Baseline 1D-CNN (Test Fold 10)", fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_p / "fig2_per_class_auroc.png", dpi=300)
    plt.close()


def plot_missing_lead_degradation(
    stgat_missing_csv: str = "runs/missing_leads/STGAT_Static/missing_lead_results.csv",
    dropout_missing_csv: str = "runs/missing_leads/STGAT_LeadDropout/missing_lead_results.csv",
    baseline_missing_csv: str = "runs/missing_leads/Baseline_CNN/missing_lead_results.csv",
    output_dir: str = "results/figures",
) -> None:
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5), dpi=300)
    for path, label, color in [
        (dropout_missing_csv, "ST-GAT (Lead-Dropout)", "#059669"),
        (stgat_missing_csv, "ST-GAT (Standard)", "#2563EB"),
        (baseline_missing_csv, "Baseline 1D-CNN", "#DC2626"),
    ]:
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            plt.plot(df["num_leads"], df["macro_auc"], marker="o", label=label, color=color, linewidth=2)

    plt.xlabel("Number of Available Leads (12 -> 1)", fontweight="bold")
    plt.ylabel("Macro AUROC", fontweight="bold")
    plt.title("Missing-Lead Robustness Degradation (RQ4 / RQ5)", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_p / "fig3_missing_lead_degradation.png", dpi=300)
    plt.close()
