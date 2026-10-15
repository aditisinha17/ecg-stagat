# Spatial-Temporal Graph Attention Network (ST-GAT) for 12-Lead ECG Classification with Interval-Level Clinical Explainability

A research experimentation and statistical validation pipeline implementing a **Multi-Task Spatial-Temporal Graph Attention Network (ST-GAT)** for multi-label 12-lead ECG classification on the **PTB-XL v1.0.3** clinical benchmark dataset.

---

## Architecture Overview

```
12-Lead ECG (10s @ 100Hz)
      │  (B, 12, 1000)
      ▼
┌────────────────────────────────────────┐
│   Signal Preprocessing Pipeline        │  0.5–45 Hz Bandpass, Baseline Removal, Z-Score
└───────────────────┬────────────────────┘
                    │  (B, 12, 1000)
                    ▼
┌────────────────────────────────────────┐
│   Per-Lead 1D-ResNet Temporal Encoder  │  Shared Residual Blocks + Conv1D + Pooling
└───────────────────┬────────────────────┘
                    │  H_temp: (B, 12, 128)
                    ├─────────────────────────────────────────┐
                    ▼                                         │  Shortcut
┌────────────────────────────────────────┐                    │
│   Spatial GATv2 Message Passing        │ ◄── Lead Graph Adjacency A (12x12)
│   (Multi-Head Dynamic Attention, K=4)  │
└───────────────────┬────────────────────┘
                    │  H_graph: (B, 12, 128)
                    ▼
┌────────────────────────────────────────┐
│   Gated Spatial-Temporal Fusion        │  H_fused = g * H_graph + (1-g) * H_temp
└───────────────────┬────────────────────┘
                    │  (B, 12, 128)
                    ▼
┌────────────────────────────────────────┐
│   Multi-Lead Attention Pooling         │  Adaptive Lead Importance Weighting
└───────────────────┬────────────────────┘
                    │  Recording Vector: (B, 128)
                    ▼
┌────────────────────────────────────────┐
│   Multi-Label Classification Head      │  Linear -> LayerNorm -> GELU -> Dropout -> 5 Logits
└───────────────────┬────────────────────┘
                    │
                    ▼
   [NORM, MI, STTC, CD, HYP] Probabilities
```

---

## Repository Structure

```
ecg_stgat/
├── data/
│   ├── preprocessing.py          # 0.5-45 Hz Bandpass filtering, baseline-wander removal, z-score norm
│   └── ptbxl_dataset.py          # WFDB loading, SCP code parsing, official 10-fold split, fast RAM cache
├── models/
│   ├── temporal_encoder.py       # 1D-ResNet & 1D-CNN per-lead feature extractors -> (B, 12, D)
│   ├── lead_graph.py             # Static anatomical, fully-connected, and learned 12x12 lead adjacency
│   ├── gat_module.py             # Multi-head GATv2 spatial message passing with attention extraction
│   ├── fusion_head.py            # Gated ST fusion + multi-lead attention pooling + classifier
│   ├── stgat_model.py            # Full ST-GAT end-to-end model wiring
│   └── baseline_cnn.py           # Non-graph 1D-CNN baseline (identical encoder & head) for H1
├── training/
│   ├── config.yaml               # Hyperparameters (learning rate, batch size, graph type, loss weights)
│   └── train.py                  # Multi-label BCE engine, mixed precision AMP, AdamW, early stopping
├── evaluation/
│   ├── metrics.py                # Macro/Micro AUROC, F1, optimal threshold search, 1000 bootstrap CIs
│   ├── evaluate.py               # Test fold (Fold 10) evaluation runner
│   ├── ablation.py               # RQ2: Representation & Graph Topology ablation experiments
│   ├── missing_lead_eval.py      # RQ4: Missing-lead evaluation across 12, 10, 8, 6, 3, 1 leads
│   └── lead_dropout_training.py  # RQ5: Training with stochastic lead-dropout regularization
├── explainability/
│   ├── saliency.py               # Gradient-based temporal saliency & SmoothGrad per lead
│   ├── lead_importance.py        # Spatial GAT attention extraction and class-level lead importance
│   ├── waveform_alignment.py     # R-peak & fiducial interval segmentation (QRS, ST, T, P) & overlap metric
│   └── run_explainability.py     # Master RQ3 explainability execution pipeline
├── diagrams/
│   ├── architecture_diagram.py   # Publication model architecture diagram generator (PNG + SVG)
│   └── dataflow_diagram.py       # End-to-end experimental data-flow diagram generator (PNG + SVG)
├── reporting/
│   ├── make_tables.py            # Generates Markdown and CSV result tables for Tables 1–5
│   └── make_plots.py             # Generates publication figures (loss curves, ROC, degradation, explainability)
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis, fold balances, signal visualization
│   └── 02_results_analysis.ipynb # Interactive results walkthrough, ROC curves, case studies
├── runs/                         # Experiment logs, checkpoints, JSON metrics, history CSVs
├── results/
│   ├── tables/                   # Formatted CSV and Markdown result tables
│   └── figures/                  # Publication figures (PNG at 300 DPI + SVG vector)
├── requirements.txt
└── README.md
```

---

## Dataset Protocol & Benchmark Setup

- **Dataset**: PTB-XL v1.0.3 (PhysioNet)
- **Target Superclasses** (from `scp_statements.csv`):
  1. `NORM`: Normal ECG (43.6% prevalence)
  2. `MI`: Myocardial Infarction (25.1% prevalence)
  3. `STTC`: ST/T-Wave Changes (24.0% prevalence)
  4. `CD`: Conduction Disturbance (22.5% prevalence)
  5. `HYP`: Hypertrophy (12.2% prevalence)
- **Official Benchmark Splits** (Strodthoff et al., 2020 protocol):
  - **Train Set**: Folds 1–8 (17,418 recordings)
  - **Validation Set**: Fold 9 (2,183 recordings)
  - **Test Set**: Fold 10 (2,198 recordings)
- **Sampling Rate**: Default 100 Hz ($T = 1000$ samples per 10-second recording).

---

## Quickstart & Reproduction Guide

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows (or source venv/bin/activate on Linux/Mac)

# Install dependencies
pip install -r requirements.txt
```

### 2. Exploratory Data Analysis & Signal Caching

```bash
# Generate EDA figures, distribution tables, and build the fast preprocessed dataset cache:
python run_eda.py
```

### 3. Run Individual Model Training

```bash
# Train Non-Graph Baseline 1D-CNN (RQ1 comparison):
python -m ecg_stgat.training.train --model baseline_cnn --exp_name Baseline_CNN

# Train Proposed ST-GAT (Static Anatomical Graph):
python -m ecg_stgat.training.train --model stgat --graph_type static --exp_name STGAT_Static

# Train ST-GAT with stochastic Lead-Dropout (RQ5):
python -m ecg_stgat.evaluation.lead_dropout_training --lead_mask_prob 0.5 --max_dropped_leads 6 --exp_name STGAT_LeadDropout
```

### 4. Run Evaluations, Benchmarks & Explainability

```bash
# Test Fold 10 Evaluation with 1000 Bootstrap Confidence Intervals:
python -m ecg_stgat.evaluation.evaluate --checkpoint runs/STGAT_Static/best_model.pt

# Run RQ2 Ablation Study (Representation, Graph Topology, Pooling):
python -m ecg_stgat.evaluation.ablation --epochs 25

# Run RQ4 Missing-Lead Robustness Benchmark (12, 10, 8, 6, 3, 1 leads):
python -m ecg_stgat.evaluation.missing_lead_eval --checkpoint runs/STGAT_Static/best_model.pt

# Run RQ3 Clinical Explainability & Interval Saliency Alignment Pipeline:
python -m ecg_stgat.explainability.run_explainability --checkpoint runs/STGAT_Static/best_model.pt
```

### 5. Run Full End-to-End Experiment Pipeline

To reproduce all experiments, tables, figures, and benchmarks in one automated run:

```bash
python run_experiments.py 25
```

---

## Research Questions & Experimental Deliverables

| RQ / Hypothesis | Topic | Deliverables & Artifacts |
|:---|:---|:---|
| **RQ1 / H1** | ST-GAT vs Baseline 1D-CNN | Table 1 (Macro/Micro AUROC, F1, 95% Bootstrap CIs), Fig 1 (Loss/AUROC Convergence Curves), Fig 2 (Per-Class Bar Chart) |
| **RQ2 / H2** | Ablation Analysis (Temporal vs Graph vs Fused, Graph Types) | Table 3 (`table3_rq2_ablation_study.md`), `runs/ablations/` |
| **RQ3 / H3** | Interval-Level Clinical Explainability | Table 5 (QRS vs ST-T vs P-wave Saliency Mass Overlap & Concentration), Fig 4–7 (`explainability_*.png`) |
| **RQ4** | Missing-Lead Robustness (12 → 1 Leads) | Table 4 (Performance degradation across lead subsets), Fig 3 (Degradation Curves) |
| **RQ5 / H4** | Lead-Dropout Training Regularization | Table 4 (Comparison with/without Lead Dropout), Fig 3 (Comparative Robustness Curves) |

---

## Key Experimental Results Summary

1. **Spatial Graph Attention Impact (RQ1/H1)**:
   - ST-GAT outperforms the non-graph Baseline 1D-CNN across all diagnostic superclasses, with highest gains observed in spatially correlated pathologies (Myocardial Infarction and Conduction Disturbances).
2. **Representation Ablation (RQ2/H2)**:
   - Gated Spatial-Temporal Fusion achieves superior performance over temporal-only and graph-only baselines, confirming that temporal morphology and inter-lead spatial message passing provide complementary clinical signals.
   - The physiologically-grounded static anatomical lead graph outperforms fully-connected graphs by constraining attention to valid anatomical lead groupings.
3. **Missing-Lead Resilience & Lead Dropout (RQ4/RQ5)**:
   - Standard models experience steep performance drops when reduced to 6, 3, or 1 lead.
   - Training with stochastic Lead-Dropout significantly mitigates degradation, maintaining high AUROC even under severely reduced lead configurations (e.g. single lead II).
4. **Clinical Interval Alignment (RQ3/H3)**:
   - Saliency maps confirm that ST-GAT places high attribution mass on pathophysiologically relevant waveform intervals: QRS complex for Conduction Disturbances (CD) and ST segment / T-wave for Myocardial Infarction (MI) and ST-T Changes (STTC).
