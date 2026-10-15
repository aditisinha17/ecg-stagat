"""
Master Experimentation Runner for ECG ST-GAT Research Pipeline.
Executes:
1. RQ1 / H1: Full Baseline 1D-CNN and ST-GAT (Static Graph) training on Folds 1-8.
2. RQ2 / H2: Ablation Experiments (Temporal-Only, Graph-Only, Concat, FC, Learned, MeanPool).
3. RQ5 / H4: ST-GAT with stochastic Lead-Dropout training.
4. RQ4: Missing-lead robustness evaluation across 12, 10, 8, 6, 3, 1 leads.
5. RQ3 / H3: Clinical Explainability & Interval Alignment pipeline.
6. Publication Figures & Result Tables generation.
"""

import os
import sys
import time
from pathlib import Path
import yaml

from ecg_stgat.training.train import train_experiment
from ecg_stgat.evaluation.evaluate import evaluate_checkpoint
from ecg_stgat.evaluation.missing_lead_eval import run_missing_lead_benchmark
from ecg_stgat.evaluation.ablation import run_ablation_suite
from ecg_stgat.evaluation.lead_dropout_training import train_lead_dropout_model
from ecg_stgat.explainability.run_explainability import run_explainability_pipeline
from ecg_stgat.reporting.make_tables import generate_all_tables
from ecg_stgat.reporting.make_plots import (
    plot_training_curves,
    plot_per_class_bar_chart,
    plot_missing_lead_degradation,
)


def run_full_experiment_pipeline(
    epochs: int = 30,
    config_path: str = "ecg_stgat/training/config.yaml",
):
    """
    Execute end-to-end experimental suite for all research questions.
    """
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    base_cfg["training"]["epochs"] = epochs
    Path("runs").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("STAGE 1: TRAINING BASELINE 1D-CNN (RQ1 Comparison Model)")
    print("="*70)
    cfg_base = yaml.safe_load(yaml.dump(base_cfg))
    cfg_base["model"]["name"] = "baseline_cnn"
    base_res = train_experiment(cfg_base, exp_name="Baseline_CNN")

    print("\n" + "="*70)
    print("STAGE 2: TRAINING ST-GAT (Proposed Model - Static Graph)")
    print("="*70)
    cfg_stgat = yaml.safe_load(yaml.dump(base_cfg))
    cfg_stgat["model"]["name"] = "stgat"
    cfg_stgat["model"]["graph_type"] = "static"
    cfg_stgat["model"]["fusion_type"] = "gated"
    stgat_res = train_experiment(cfg_stgat, exp_name="STGAT_Static")

    # Evaluate with 1000 Bootstrap CIs
    print("\n[Evaluation] Running 1000 Bootstrap Resamples on Test Set for RQ1...")
    evaluate_checkpoint("runs/Baseline_CNN/best_model.pt", bootstrap=True, bootstrap_iterations=1000)
    evaluate_checkpoint("runs/STGAT_Static/best_model.pt", bootstrap=True, bootstrap_iterations=1000)

    print("\n" + "="*70)
    print("STAGE 3: RUNNING RQ2 ABLATION EXPERIMENTS")
    print("="*70)
    run_ablation_suite(base_config_path=config_path, epochs=epochs, output_dir="runs/ablations")

    print("\n" + "="*70)
    print("STAGE 4: TRAINING ST-GAT WITH STOCHASTIC LEAD DROPOUT (RQ5 / H4)")
    print("="*70)
    dropout_res = train_lead_dropout_model(
        config_path=config_path,
        lead_mask_prob=0.5,
        max_dropped_leads=6,
        exp_name="STGAT_LeadDropout",
        epochs=epochs,
    )

    print("\n" + "="*70)
    print("STAGE 5: RQ4 & RQ5 MISSING-LEAD ROBUSTNESS BENCHMARKS (12 -> 1 Leads)")
    print("="*70)
    print("Benchmarking Baseline CNN...")
    run_missing_lead_benchmark("runs/Baseline_CNN/best_model.pt", output_dir="runs/missing_leads")

    print("Benchmarking ST-GAT (Standard)...")
    run_missing_lead_benchmark("runs/STGAT_Static/best_model.pt", output_dir="runs/missing_leads")

    print("Benchmarking ST-GAT (Lead-Dropout)...")
    run_missing_lead_benchmark("runs/STGAT_LeadDropout/best_model.pt", output_dir="runs/missing_leads")

    print("\n" + "="*70)
    print("STAGE 6: RQ3 CLINICAL EXPLAINABILITY & INTERVAL ALIGNMENT PIPELINE")
    print("="*70)
    run_explainability_pipeline(
        checkpoint_path="runs/STGAT_Static/best_model.pt",
        figures_dir="results/figures",
        tables_dir="results/tables",
    )

    print("\n" + "="*70)
    print("STAGE 7: GENERATING ALL REPORT TABLES & PUBLICATION FIGURES")
    print("="*70)
    generate_all_tables(
        stgat_exp_dir="runs/STGAT_Static",
        baseline_exp_dir="runs/Baseline_CNN",
        dropout_exp_dir="runs/STGAT_LeadDropout",
        ablations_dir="runs/ablations",
        missing_leads_dir="runs/missing_leads",
        output_dir="results/tables",
    )

    plot_training_curves(
        stgat_history_csv="runs/STGAT_Static/training_history.csv",
        baseline_history_csv="runs/Baseline_CNN/training_history.csv",
        output_dir="results/figures",
    )

    plot_per_class_bar_chart(
        stgat_metrics_json="runs/STGAT_Static/metrics.json",
        baseline_metrics_json="runs/Baseline_CNN/metrics.json",
        output_dir="results/figures",
    )

    plot_missing_lead_degradation(
        stgat_missing_csv="runs/missing_leads/STGAT_Static/missing_lead_results.csv",
        dropout_missing_csv="runs/missing_leads/STGAT_LeadDropout/missing_lead_results.csv",
        baseline_missing_csv="runs/missing_leads/Baseline_CNN/missing_lead_results.csv",
        output_dir="results/figures",
    )

    print("\n" + "="*70)
    print(">>> ALL EXPERIMENTAL PIPELINES COMPLETED SUCCESSFULLY! <<<")
    print("All result tables saved in results/tables/")
    print("All publication figures saved in results/figures/")
    print("="*70)


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_full_experiment_pipeline(epochs=epochs)
