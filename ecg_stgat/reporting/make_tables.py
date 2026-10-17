"""
Generates formatted Markdown and CSV tables for research reports and publication.
Tables 1–5:
- Table 1: RQ1 / H1 Model Performance (Macro/Micro AUROC, F1, 95% CIs, Exact Match)
- Table 2: Diagnostic Superclass Per-Class Breakdown (AUROC & F1 per class)
- Table 3: RQ2 Ablation Study (Representation, Graph Topology, Pooling)
- Table 4: RQ4/RQ5 Missing-Lead Robustness Across 12, 10, 8, 6, 3, 1 leads
- Table 5: RQ3 Interval Saliency Alignment Summary
"""
from typing import Optional, Dict, Any
import json
from pathlib import Path
import pandas as pd


def load_json_safe(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def generate_all_tables(
    stgat_exp_dir: str = "runs/STGAT_Static",
    baseline_exp_dir: str = "runs/Baseline_CNN",
    dropout_exp_dir: str = "runs/STGAT_LeadDropout",
    ablations_dir: str = "runs/ablations",
    missing_leads_dir: str = "runs/missing_leads",
    output_dir: str = "results/tables",
) -> Dict[str, pd.DataFrame]:
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    generated_tables = {}

    # -------------------------------------------------------------
    # Table 1: RQ1 Model Comparison with 95% Bootstrap CIs
    # -------------------------------------------------------------
    stgat_m = load_json_safe(Path(stgat_exp_dir) / "test_metrics.json")
    base_m = load_json_safe(Path(baseline_exp_dir) / "test_metrics.json")

    def format_ci(val: float, ci: Optional[list]) -> str:
        if ci and len(ci) == 2:
            return f"{val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        return f"{val:.4f}"

    t1_rows = []
    if base_m:
        t1_rows.append({
            "Model Architecture": "Baseline 1D-CNN",
            "Spatial Graph Topology": "None (Non-Graph)",
            "Macro AUROC (95% CI)": format_ci(base_m["macro_auc"], base_m.get("macro_auc_ci")),
            "Micro AUROC": f"{base_m['micro_auc']:.4f}",
            "Macro F1 (95% CI)": format_ci(base_m["macro_f1"], base_m.get("macro_f1_ci")),
            "Micro F1": f"{base_m['micro_f1']:.4f}",
            "Exact Match Acc": f"{base_m.get('exact_match_accuracy', 0.0):.4f}",
        })

    if stgat_m:
        t1_rows.append({
            "Model Architecture": "ST-GAT (Proposed)",
            "Spatial Graph Topology": "Static Clinically Motivated",
            "Macro AUROC (95% CI)": format_ci(stgat_m["macro_auc"], stgat_m.get("macro_auc_ci")),
            "Micro AUROC": f"{stgat_m['micro_auc']:.4f}",
            "Macro F1 (95% CI)": format_ci(stgat_m["macro_f1"], stgat_m.get("macro_f1_ci")),
            "Micro F1": f"{stgat_m['micro_f1']:.4f}",
            "Exact Match Acc": f"{stgat_m.get('exact_match_accuracy', 0.0):.4f}",
        })

    if t1_rows:
        df_t1 = pd.DataFrame(t1_rows)
        df_t1.to_csv(out_p / "table1_rq1_model_performance.csv", index=False)
        df_t1.to_markdown(out_p / "table1_rq1_model_performance.md", index=False)
        generated_tables["Table 1"] = df_t1

    # -------------------------------------------------------------
    # Table 2: Diagnostic Superclass Per-Class Breakdown
    # -------------------------------------------------------------
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    class_full_names = {
        "NORM": "Normal ECG (NORM)",
        "MI": "Myocardial Infarction (MI)",
        "STTC": "ST/T-Wave Changes (STTC)",
        "CD": "Conduction Disturbance (CD)",
        "HYP": "Hypertrophy (HYP)",
    }

    if base_m and stgat_m:
        t2_rows = []
        base_auc_dict = base_m.get("per_class_auc", {})
        stgat_auc_dict = stgat_m.get("per_class_auc", {})
        base_f1_dict = base_m.get("per_class_f1", {})
        stgat_f1_dict = stgat_m.get("per_class_f1", {})

        for c in classes:
            b_auc = base_auc_dict.get(c, 0.0)
            s_auc = stgat_auc_dict.get(c, 0.0)
            b_f1 = base_f1_dict.get(c, 0.0)
            s_f1 = stgat_f1_dict.get(c, 0.0)

            t2_rows.append({
                "Superclass": class_full_names.get(c, c),
                "Baseline AUROC": f"{b_auc:.4f}",
                "ST-GAT AUROC": f"{s_auc:.4f}",
                "Delta AUROC": f"{(s_auc - b_auc):+.4f}",
                "Baseline F1": f"{b_f1:.4f}",
                "ST-GAT F1": f"{s_f1:.4f}",
                "Delta F1": f"{(s_f1 - b_f1):+.4f}",
            })

        df_t2 = pd.DataFrame(t2_rows)
        df_t2.to_csv(out_p / "table2_per_class_breakdown.csv", index=False)
        df_t2.to_markdown(out_p / "table2_per_class_breakdown.md", index=False)
        generated_tables["Table 2"] = df_t2

    # -------------------------------------------------------------
    # Table 3: RQ2 Ablation Study
    # -------------------------------------------------------------
    ablation_p = Path(ablations_dir)
    ablation_rows = []

    # Include Baseline and Full ST-GAT as references
    if base_m:
        ablation_rows.append({
            "Ablation Condition": "Baseline 1D-CNN (Reference)",
            "Component Modified": "No spatial graph (temporal encoder only)",
            "Macro AUROC": f"{base_m['macro_auc']:.4f}",
            "Micro AUROC": f"{base_m['micro_auc']:.4f}",
            "Macro F1": f"{base_m['macro_f1']:.4f}",
            "Micro F1": f"{base_m['micro_f1']:.4f}",
            "Exact Match Acc": f"{base_m.get('exact_match_accuracy', 0.0):.4f}",
        })

    ablation_descriptions = {
        "temporal_only": "Temporal encoder only (bypasses spatial graph)",
        "graph_only": "Graph attention only (bypasses temporal shortcut)",
        "concat_fusion": "Concatenation fusion instead of gated fusion",
        "fc_graph": "Fully-connected dense graph (all-to-all control topology)",
        "mean_pooling": "Uniform mean pooling instead of attention pooling",
    }

    # Load from ablation subdirectories
    for cond_name, desc in ablation_descriptions.items():
        cond_m = load_json_safe(ablation_p / cond_name / "test_metrics.json")
        if cond_m:
            ablation_rows.append({
                "Ablation Condition": cond_name.replace("_", " ").title(),
                "Component Modified": desc,
                "Macro AUROC": f"{cond_m['macro_auc']:.4f}",
                "Micro AUROC": f"{cond_m['micro_auc']:.4f}",
                "Macro F1": f"{cond_m['macro_f1']:.4f}",
                "Micro F1": f"{cond_m['micro_f1']:.4f}",
                "Exact Match Acc": f"{cond_m.get('exact_match_accuracy', 0.0):.4f}",
            })

    if stgat_m:
        ablation_rows.append({
            "Ablation Condition": "Full ST-GAT (Proposed)",
            "Component Modified": "Gated fusion + Static Lead Graph + Attention Pooling",
            "Macro AUROC": f"{stgat_m['macro_auc']:.4f}",
            "Micro AUROC": f"{stgat_m['micro_auc']:.4f}",
            "Macro F1": f"{stgat_m['macro_f1']:.4f}",
            "Micro F1": f"{stgat_m['micro_f1']:.4f}",
            "Exact Match Acc": f"{stgat_m.get('exact_match_accuracy', 0.0):.4f}",
        })

    if ablation_rows:
        df_t3 = pd.DataFrame(ablation_rows)
        df_t3.to_csv(out_p / "table3_rq2_ablation_study.csv", index=False)
        df_t3.to_markdown(out_p / "table3_rq2_ablation_study.md", index=False)
        generated_tables["Table 3"] = df_t3

    # -------------------------------------------------------------
    # Table 4: RQ4 / RQ5 Missing-Lead Robustness Across Lead Subsets
    # -------------------------------------------------------------
    miss_dir = Path(missing_leads_dir)
    stgat_miss = miss_dir / "STGAT_Static" / "missing_lead_results.csv"
    dropout_miss = miss_dir / "STGAT_LeadDropout" / "missing_lead_results.csv"
    base_miss = miss_dir / "Baseline_CNN" / "missing_lead_results.csv"

    if stgat_miss.exists() or base_miss.exists():
        t4_dfs = []
        if base_miss.exists():
            df_b = pd.read_csv(base_miss)
            df_b["Model"] = "Baseline 1D-CNN"
            t4_dfs.append(df_b)
        if stgat_miss.exists():
            df_s = pd.read_csv(stgat_miss)
            df_s["Model"] = "ST-GAT (Standard)"
            t4_dfs.append(df_s)
        if dropout_miss.exists():
            df_d = pd.read_csv(dropout_miss)
            df_d["Model"] = "ST-GAT (Lead-Dropout)"
            t4_dfs.append(df_d)

        if t4_dfs:
            combined_miss = pd.concat(t4_dfs, axis=0)
            pivot_auc = combined_miss.pivot(index="num_leads", columns="Model", values="macro_auc").sort_index(ascending=False)
            pivot_f1 = combined_miss.pivot(index="num_leads", columns="Model", values="macro_f1").sort_index(ascending=False)

            pivot_auc.to_csv(out_p / "table4_rq4_missing_lead_robustness.csv")
            pivot_auc.to_markdown(out_p / "table4_rq4_missing_lead_robustness.md")
            generated_tables["Table 4"] = pivot_auc

    print(f"[Reporting] Tables generated in {output_dir}: {list(generated_tables.keys())}")
    return generated_tables
