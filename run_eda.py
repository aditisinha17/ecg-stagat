"""
Comprehensive Exploratory Data Analysis (EDA) Script and Figure Generator.
Generates:
1. Class distribution bar charts and imbalance statistics
2. Multi-label co-occurrence heatmap
3. Stratified fold split balance verification
4. Representative 12-lead ECG signal plots across diagnostic superclasses
5. Lead amplitude and signal statistics
Saves all figures into results/figures/ for the final research report.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecg_stgat.data.ptbxl_dataset import (
    load_raw_ptbxl_metadata,
    map_scp_to_superclasses,
    DIAGNOSTIC_CLASSES,
    LEAD_NAMES,
    prepare_cached_ptbxl_data,
)


def run_eda(
    data_dir: str = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    figures_dir: str = "results/figures",
    tables_dir: str = "results/tables",
) -> Dict[str, Any]:
    """
    Run full Exploratory Data Analysis on PTB-XL v1.0.3 and save report artifacts.
    """
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

    # 1. Load Metadata
    print("[EDA] Loading PTB-XL metadata...")
    df, diag_scp = load_raw_ptbxl_metadata(data_dir)
    df["diagnostic_superclass"] = df["scp_codes_dict"].apply(
        lambda d: map_scp_to_superclasses(d, diag_scp)
    )

    num_records = len(df)
    num_patients = df["patient_id"].nunique()
    print(f"[EDA] Total recordings: {num_records:,} | Unique patients: {num_patients:,}")

    # Build binary indicator columns for each diagnostic class
    for cls_name in DIAGNOSTIC_CLASSES:
        df[cls_name] = df["diagnostic_superclass"].apply(lambda classes: int(cls_name in classes))

    df["num_positive_labels"] = df[DIAGNOSTIC_CLASSES].sum(axis=1)

    # 2. Class Counts and Imbalance Ratios
    class_counts = df[DIAGNOSTIC_CLASSES].sum().to_dict()
    class_pcts = {k: (v / num_records) * 100.0 for k, v in class_counts.items()}
    imbalance_df = pd.DataFrame({
        "Superclass": DIAGNOSTIC_CLASSES,
        "Positive Counts": [class_counts[c] for c in DIAGNOSTIC_CLASSES],
        "Prevalence (%)": [f"{class_pcts[c]:.2f}%" for c in DIAGNOSTIC_CLASSES],
        "Imbalance Ratio (vs Min)": [f"{class_counts[c] / min(class_counts.values()):.2f}x" for c in DIAGNOSTIC_CLASSES],
    })
    imbalance_df.to_csv(Path(tables_dir) / "eda_class_distribution.csv", index=False)
    imbalance_df.to_markdown(Path(tables_dir) / "eda_class_distribution.md", index=False)

    print("\n--- Diagnostic Superclass Distribution ---")
    print(imbalance_df.to_string(index=False))

    # Figure 1: Class Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    palette = ["#2563EB", "#DC2626", "#D97706", "#7C3AED", "#059669"]
    bars = ax.bar(DIAGNOSTIC_CLASSES, [class_counts[c] for c in DIAGNOSTIC_CLASSES], color=palette, width=0.55, edgecolor="black", linewidth=0.8)
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / num_records) * 100.0
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 150, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(class_counts.values()) * 1.18)
    ax.set_title("PTB-XL v1.0.3 Diagnostic Superclass Distribution (N = 21,837)", fontweight="bold", pad=12)
    ax.set_xlabel("Diagnostic Superclass", fontweight="bold")
    ax.set_ylabel("Number of Recordings", fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(figures_dir) / "eda_class_distribution.png", dpi=300)
    plt.savefig(Path(figures_dir) / "eda_class_distribution.svg", format="svg")
    plt.close()

    # 3. Stratified Fold Split Verification
    fold_summary = []
    for fold in range(1, 11):
        fold_df = df[df["strat_fold"] == fold]
        role = "Train (1-8)" if fold <= 8 else ("Val (9)" if fold == 9 else "Test (10)")
        row = {"Fold": fold, "Role": role, "Total": len(fold_df)}
        for c in DIAGNOSTIC_CLASSES:
            row[c] = fold_df[c].sum()
        fold_summary.append(row)

    fold_df = pd.DataFrame(fold_summary)
    fold_df.to_csv(Path(tables_dir) / "eda_fold_distribution.csv", index=False)
    fold_df.to_markdown(Path(tables_dir) / "eda_fold_distribution.md", index=False)

    # 4. Multi-Label Co-occurrence Matrix
    cooc_matrix = np.zeros((len(DIAGNOSTIC_CLASSES), len(DIAGNOSTIC_CLASSES)), dtype=int)
    for i, c1 in enumerate(DIAGNOSTIC_CLASSES):
        for j, c2 in enumerate(DIAGNOSTIC_CLASSES):
            cooc_matrix[i, j] = int((df[c1] & df[c2]).sum())

    cooc_df = pd.DataFrame(cooc_matrix, index=DIAGNOSTIC_CLASSES, columns=DIAGNOSTIC_CLASSES)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.heatmap(cooc_df, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax, linewidths=0.8, linecolor="white", annot_kws={"size": 10, "weight": "bold"})
    ax.set_title("Diagnostic Superclass Co-occurrence Matrix (PTB-XL)", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(Path(figures_dir) / "eda_label_cooccurrence.png", dpi=300)
    plt.savefig(Path(figures_dir) / "eda_label_cooccurrence.svg", format="svg")
    plt.close()

    # 5. Signal Loading and Representative 12-Lead ECG Plots
    print("\n[EDA] Preparing signal cache and representative waveform plots...")
    signals, labels, meta_df = prepare_cached_ptbxl_data(
        data_dir=data_dir,
        sampling_rate=100,
    )

    # Plot representative 12-lead ECG for each diagnostic superclass
    for cls_name in DIAGNOSTIC_CLASSES:
        cls_idx = DIAGNOSTIC_CLASSES.index(cls_name)
        match_idx = np.where((labels[:, cls_idx] == 1.0) & (labels.sum(axis=1) == 1.0))[0]
        if len(match_idx) == 0:
            match_idx = np.where(labels[:, cls_idx] == 1.0)[0]
        sample_idx = match_idx[0]
        sample_ecg = signals[sample_idx]  # (12, 1000)
        ecg_id = meta_df.index[sample_idx]

        fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True, dpi=300)
        fig.suptitle(f"Representative 12-Lead ECG: {cls_name} (ECG ID: {ecg_id}, 10s @ 100Hz)", fontsize=13, fontweight="bold", y=0.99)
        time_axis = np.linspace(0, 10, 1000)

        for lead_i, lead_name in enumerate(LEAD_NAMES):
            row_i = lead_i % 6
            col_i = lead_i // 6
            ax_lead = axes[row_i, col_i]
            ax_lead.plot(time_axis, sample_ecg[lead_i], color="#1E3A8A", linewidth=1.0)
            ax_lead.set_ylabel(lead_name, fontweight="bold", fontsize=10, rotation=0, labelpad=15)
            ax_lead.grid(True, linestyle="--", alpha=0.5)
            if row_i == 5:
                ax_lead.set_xlabel("Time (seconds)", fontsize=9.5)

        plt.tight_layout()
        plt.savefig(Path(figures_dir) / f"eda_waveform_{cls_name.lower()}.png", dpi=300)
        plt.close()

    eda_summary = {
        "total_recordings": int(num_records),
        "total_patients": int(num_patients),
        "train_samples": int((df["strat_fold"] <= 8).sum()),
        "val_samples": int((df["strat_fold"] == 9).sum()),
        "test_samples": int((df["strat_fold"] == 10).sum()),
        "class_counts": class_counts,
        "class_prevalence_pct": class_pcts,
        "single_label_recordings": int((df["num_positive_labels"] == 1).sum()),
        "multi_label_recordings": int((df["num_positive_labels"] > 1).sum()),
        "zero_label_recordings": int((df["num_positive_labels"] == 0).sum()),
    }

    with open(Path(tables_dir) / "eda_summary.json", "w") as f:
        json.dump(eda_summary, f, indent=2)

    print("\n[EDA Completed Successfully] All summary tables and figures generated.")
    return eda_summary


if __name__ == "__main__":
    run_eda()
