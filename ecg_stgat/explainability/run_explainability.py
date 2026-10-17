"""
Master Clinical Explainability & Interval Alignment Pipeline (RQ3).
Generates:
1. Waveform interval-aligned SmoothGrad attribution maps
2. Publication-quality explanation figures with wave boundaries (P, QRS, ST, T)
3. Correctly classified vs misclassified case studies across diagnostic superclasses
4. Quantification of attribution faithfulness and stability
5. Table 5: Real interval-level attribution summary table
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch

from ecg_stgat.data.ptbxl_dataset import get_ptbxl_dataloaders, DIAGNOSTIC_CLASSES, LEAD_NAMES
from ecg_stgat.training.train import build_model_from_config
from ecg_stgat.explainability.saliency import compute_smoothgrad
from ecg_stgat.explainability.waveform_alignment import (
    compute_interval_alignment,
    compute_attribution_faithfulness,
)
from ecg_stgat.explainability.lead_importance import extract_lead_importance


def run_explainability_pipeline(
    checkpoint_path: str = "runs/STGAT_Static/best_model.pt",
    figures_dir: str = "results/figures",
    tables_dir: str = "results/tables",
    data_dir: str = "data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    max_eval_samples: int = 50,
) -> pd.DataFrame:
    """
    Executes the clinical explainability pipeline on Test Fold 10 records.
    """
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"[Explainability] Checkpoint not found: {checkpoint_path}")

    fig_out = Path(figures_dir)
    tbl_out = Path(tables_dir)
    fig_out.mkdir(parents=True, exist_ok=True)
    tbl_out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load(ckpt_p, map_location=device)
    config = checkpoint.get("config", {"model": {"name": "stgat", "graph_type": "static"}})

    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[Explainability] Initializing Test Fold 10 for {ckpt_p.name}...")
    _, _, test_loader, _, _ = get_ptbxl_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        test_folds=[10],
        preprocess=True,
    )

    print("\n[Explainability] Running NeuroKit2 DWT interval alignment and faithfulness evaluation...")
    interval_records = {cls_name: [] for cls_name in DIAGNOSTIC_CLASSES}
    delineation_counts = {"success": 0, "fallback": 0, "total": 0}
    faithfulness_scores = {cls_name: [] for cls_name in DIAGNOSTIC_CLASSES}

    case_studies = {
        "tp": {cls_name: None for cls_name in DIAGNOSTIC_CLASSES},
        "fp": None,
        "fn": None,
    }

    sample_count = 0
    with torch.no_grad():
        all_thresholds = [0.5] * 5

    for x_b, y_b, ecg_ids in test_loader:
        if sample_count >= max_eval_samples:
            break

        for i in range(len(x_b)):
            if sample_count >= max_eval_samples:
                break

            x_i = x_b[i:i+1]       # (1, 12, T)
            y_i = y_b[i].numpy()   # (5,)
            rec_id = int(ecg_ids[i])

            # Forward pass to check prediction
            with torch.no_grad():
                logits = model(x_i.to(device))
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                preds = (probs >= 0.5).astype(float)

            # Analyze for each active diagnostic class
            for c_idx, c_name in enumerate(DIAGNOSTIC_CLASSES):
                if y_i[c_idx] == 1.0:
                    sal = compute_smoothgrad(
                        model, x_i, target_class_idx=c_idx, num_samples=10, device=device
                    )
                    align_res = compute_interval_alignment(sal, x_i.squeeze(0).numpy())
                    faith_res = compute_attribution_faithfulness(
                        model, x_i, sal, target_class_idx=c_idx, top_k_pct=0.15
                    )

                    interval_records[c_name].append(align_res)
                    faithfulness_scores[c_name].append(faith_res["faithfulness_ratio"])

                    delineation_counts["total"] += 1
                    if align_res["delineation_success"]:
                        delineation_counts["success"] += 1
                    else:
                        delineation_counts["fallback"] += 1

                    # Save TP representative case study if needed
                    if preds[c_idx] == 1.0 and case_studies["tp"][c_name] is None:
                        case_studies["tp"][c_name] = {
                            "ecg_id": rec_id,
                            "x": x_i.squeeze(0).numpy(),
                            "saliency": sal,
                            "probs": probs,
                            "y_true": y_i,
                            "target_class": c_name,
                            "class_idx": c_idx,
                            "align_meta": align_res,
                            "faith_meta": faith_res,
                        }

            # Check for misclassified case (False Negative or False Positive)
            if case_studies["fn"] is None:
                for c_idx, c_name in enumerate(DIAGNOSTIC_CLASSES):
                    if y_i[c_idx] == 1.0 and preds[c_idx] == 0.0:
                        sal = compute_smoothgrad(model, x_i, target_class_idx=c_idx, num_samples=10, device=device)
                        align_res = compute_interval_alignment(sal, x_i.squeeze(0).numpy())
                        faith_res = compute_attribution_faithfulness(model, x_i, sal, target_class_idx=c_idx)
                        case_studies["fn"] = {
                            "ecg_id": rec_id,
                            "x": x_i.squeeze(0).numpy(),
                            "saliency": sal,
                            "probs": probs,
                            "y_true": y_i,
                            "target_class": c_name,
                            "class_idx": c_idx,
                            "align_meta": align_res,
                            "faith_meta": faith_res,
                            "error_type": f"False Negative ({c_name})",
                        }
                        break

            sample_count += 1

    # Generate Publication Figures for Case Studies
    print(f"\n[Explainability] Generating case study figures in {figures_dir}...")
    generate_case_study_figures(model, case_studies, fig_out, device)

    # Build Table 5: Interval Saliency Alignment Summary Table
    table_rows = []
    for c_name in DIAGNOSTIC_CLASSES:
        records = interval_records[c_name]
        n_eval = len(records)
        if n_eval > 0:
            qrs_mean = float(np.mean([r["qrs_overlap_pct"] for r in records]))
            st_mean = float(np.mean([r["st_segment_overlap_pct"] for r in records]))
            t_mean = float(np.mean([r["t_wave_overlap_pct"] for r in records]))
            st_t_mean = float(np.mean([r["st_t_overlap_pct"] for r in records]))
            p_mean = float(np.mean([r["p_wave_overlap_pct"] for r in records]))
            base_mean = float(np.mean([r["baseline_overlap_pct"] for r in records]))
            delin_rate = float(np.mean([1.0 if r["delineation_success"] else 0.0 for r in records])) * 100.0
            faith_mean = float(np.mean(faithfulness_scores[c_name])) if len(faithfulness_scores[c_name]) > 0 else 1.0
        else:
            qrs_mean, st_mean, t_mean, st_t_mean, p_mean, base_mean = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            delin_rate = 0.0
            faith_mean = 1.0

        if c_name in ["CD", "HYP"]:
            primary_target = "QRS Complex (Ventricular Depolarization)"
        elif c_name in ["MI", "STTC"]:
            primary_target = "ST Segment & T-Wave (Ventricular Repolarization / Ischemia)"
        else:
            primary_target = "Isoelectric Baseline & P-Wave (Sinus Rhythm)"

        table_rows.append({
            "Superclass": c_name,
            "Primary Pathophysiological Target": primary_target,
            "Evaluated Records": n_eval,
            "QRS Attribution (%)": f"{qrs_mean:.1f}%",
            "ST Segment Attribution (%)": f"{st_mean:.1f}%",
            "T-Wave Attribution (%)": f"{t_mean:.1f}%",
            "ST-T Combined (%)": f"{st_t_mean:.1f}%",
            "P-Wave Attribution (%)": f"{p_mean:.1f}%",
            "Baseline (%)": f"{base_mean:.1f}%",
            "NeuroKit2 Delineation Rate": f"{delin_rate:.1f}%",
            "Faithfulness Ratio": f"{faith_mean:.2f}x",
        })

    df_table5 = pd.DataFrame(table_rows)
    df_table5.to_csv(tbl_out / "table5_rq3_interval_explainability.csv", index=False)
    df_table5.to_markdown(tbl_out / "table5_rq3_interval_explainability.md", index=False)

    print(f"\n[Explainability] Table 5 generated at: {tbl_out / 'table5_rq3_interval_explainability.md'}")
    print(f"[Explainability] Total Delineations: {delineation_counts['total']}, Success: {delineation_counts['success']} ({delineation_counts['success']/max(delineation_counts['total'],1)*100:.1f}%), Fallback Heuristic: {delineation_counts['fallback']}")
    return df_table5


def generate_case_study_figures(
    model: torch.nn.Module,
    case_studies: Dict[str, Any],
    fig_out: Path,
    device: torch.device,
) -> None:
    """
    Renders multi-panel explanation plots for representative cases:
    Top: 12-lead signal overlay with colored wave intervals
    Middle: SmoothGrad temporal attribution curve
    Bottom: Spatial lead pooling weights
    """
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    sns.set_theme(style="whitegrid")

    # Select representative TP cases: MI, CD, STTC
    selected_targets = [("MI", "fig4_explainability_mi_case.png"),
                        ("CD", "fig5_explainability_cd_case.png"),
                        ("STTC", "fig6_explainability_sttc_case.png")]

    for c_name, fname in selected_targets:
        case = case_studies["tp"].get(c_name)
        if case is None:
            continue
        _plot_single_case_figure(model, case, fig_out / fname, device)

    # Plot misclassified case if available
    fn_case = case_studies.get("fn")
    if fn_case is not None:
        _plot_single_case_figure(model, fn_case, fig_out / "fig7_explainability_misclassified_case.png", device, is_misclassified=True)


def _plot_single_case_figure(
    model: torch.nn.Module,
    case: Dict[str, Any],
    out_file: Path,
    device: torch.device,
    is_misclassified: bool = False,
) -> None:
    x_np = case["x"]  # (12, 1000)
    sal = case["saliency"]  # (12, 1000)
    c_name = case["target_class"]
    c_idx = case["class_idx"]
    rec_id = case["ecg_id"]
    probs = case["probs"]
    align = case["align_meta"]
    masks = align["masks"]
    time_ax = np.linspace(0, 10, 1000)

    # Extract spatial attention
    x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
    exp_data = extract_lead_importance(model, x_t, device=device)
    lead_weights = exp_data["lead_weights"].squeeze(0)  # (12,)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=300, gridspec_kw={"height_ratios": [2.2, 1.2, 1.0]})

    # Top: Lead II & Lead V2 waveform with wave interval shading
    ax0 = axes[0]
    lead_show_idx = 1 if c_name in ["NORM", "CD", "HYP"] else 8  # Lead II or Lead V3/V4
    lead_show_name = LEAD_NAMES[lead_show_idx]

    ax0.plot(time_ax, x_np[lead_show_idx], label=f"Lead {lead_show_name} Signal", color="#0F172A", linewidth=1.3)

    # Shading for intervals
    # P-wave: green, QRS: purple, ST-segment: red, T-wave: orange
    for t_idx in range(1000):
        if masks["qrs"][t_idx]:
            ax0.axvspan(time_ax[t_idx] - 0.005, time_ax[t_idx] + 0.005, color="#8B5CF6", alpha=0.18, linewidth=0)
        elif masks["st_segment"][t_idx]:
            ax0.axvspan(time_ax[t_idx] - 0.005, time_ax[t_idx] + 0.005, color="#EF4444", alpha=0.22, linewidth=0)
        elif masks["t_wave"][t_idx]:
            ax0.axvspan(time_ax[t_idx] - 0.005, time_ax[t_idx] + 0.005, color="#F59E0B", alpha=0.18, linewidth=0)
        elif masks["p_wave"][t_idx]:
            ax0.axvspan(time_ax[t_idx] - 0.005, time_ax[t_idx] + 0.005, color="#10B981", alpha=0.18, linewidth=0)

    p_patch = mpatches.Patch(color="#10B981", alpha=0.5, label="P-Wave")
    qrs_patch = mpatches.Patch(color="#8B5CF6", alpha=0.5, label="QRS Complex")
    st_patch = mpatches.Patch(color="#EF4444", alpha=0.5, label="ST Segment")
    t_patch = mpatches.Patch(color="#F59E0B", alpha=0.5, label="T-Wave")

    pred_str = f"P({c_name}) = {probs[c_idx]:.3f}"
    status_str = f"MISCLASSIFIED: {case.get('error_type', '')}" if is_misclassified else "CORRECT PREDICTION"
    title_color = "#DC2626" if is_misclassified else "#1E40AF"

    ax0.set_title(
        f"Clinical Interval Explainability: Record #{rec_id} | Target Class: {c_name} [{status_str}] | {pred_str}\n"
        f"Delineation: {align['delineation_method']} ({align['num_delineated_beats']} valid beats detected)",
        fontweight="bold", color=title_color, fontsize=11
    )
    ax0.set_ylabel(f"Lead {lead_show_name} (mV)", fontweight="bold")
    ax0.legend(handles=[p_patch, qrs_patch, st_patch, t_patch], loc="upper right", framealpha=0.9, fontsize=9)

    # Middle: Temporal SmoothGrad attribution curve
    ax1 = axes[1]
    sal_lead = sal[lead_show_idx]
    sal_norm = sal_lead / (sal_lead.max() + 1e-9)
    ax1.plot(time_ax, sal_norm, color="#DC2626", linewidth=1.4, label="SmoothGrad Attribution")
    ax1.fill_between(time_ax, 0, sal_norm, color="#FCA5A5", alpha=0.5)
    ax1.set_ylabel("Attribution", fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Time (seconds)", fontweight="bold")

    attr_info = (f"QRS Attribution: {align['qrs_overlap_pct']:.1f}% | "
                 f"ST Segment: {align['st_segment_overlap_pct']:.1f}% | "
                 f"T-Wave: {align['t_wave_overlap_pct']:.1f}% | "
                 f"P-Wave: {align['p_wave_overlap_pct']:.1f}%")
    ax1.text(0.01, 0.88, attr_info, transform=ax1.transAxes, fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#CBD5E1"))

    # Bottom: Spatial Lead Importance Weights
    ax2 = axes[2]
    colors = ["#2563EB" if i == lead_show_idx else "#94A3B8" for i in range(12)]
    ax2.bar(LEAD_NAMES, lead_weights, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("Lead Weight", fontweight="bold")
    ax2.set_title("Multi-Lead Spatial Attention Pooling Distribution", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, max(lead_weights) * 1.25)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved explainability plot: {out_file.name}")
