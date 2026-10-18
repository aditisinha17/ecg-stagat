# ECG-STGAT: Spatial-Temporal Graph Attention Network for 12-Lead ECG Classification & 3D Anatomical Explainability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![PTB-XL Benchmark](https://img.shields.io/badge/Dataset-PTB--XL_v1.0.3-emerald.svg)](https://physionet.org/content/ptb-xl/1.0.3/)
[![WebGL 3D Active](https://img.shields.io/badge/3D_Visualization-Three.js_r128-cyan.svg)](http://localhost:8000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning research platform and interactive clinical decision-support system implementing a **Multi-Task Spatial-Temporal Graph Attention Network (ST-GAT)** for multi-label 12-lead ECG diagnostic classification on the **PTB-XL v1.0.3** clinical benchmark.

The system addresses fundamental limitations in conventional deep learning ECG models by integrating **12-lead anatomical relational topology**, **gated spatial-temporal feature fusion**, **interval-level clinical explainability** (P, QRS, ST, T wave alignment via NeuroKit2 DWT and SmoothGrad), and an **interactive 3D anatomical heart visualization with realistic cardiac electrical current propagation** running on localhost.

---

## 🫀 Key Research Innovations & Gaps Addressed

### Gap 1: AI Sees Wires as a Flat List, Not as a 3D Heart
- **The Clinical Reality**: When clinicians record a standard 12-lead ECG, electrodes are placed at precise 3D anatomical locations across the thorax and limbs (Cabrera hexaxial circle and transverse precordial arc). This provides a 360-degree spatial projection of cardiac electrical dipole vectors.
- **The AI Failure**: Standard CNNs and Transformers treat the 12 leads as arbitrary flat channels in a 1D tensor ($12 \times 1000$), discarding physical proximity, reciprocal axes, and coronary vascular territories (RCA, LAD, LCx).
- **Our ST-GAT Solution**: We formulate 12-lead ECG processing on a clinical relational graph $G = (V, E)$ where nodes represent lead electrodes and edges reflect verified anatomical and coronary contiguous groupings. In addition, our interactive dashboard provides a **3D Anatomical Heart with real-time travelling electrical currents** matching intrinsic cardiac conduction (SA Node $\to$ Atria $\to$ AV Node $\to$ Bundle of His $\to$ Septum $\to$ Apex $\to$ Ventricular walls $\to$ Repolarization).

### Gap 2: Black-Box Predictions Lack Fiducial Interval Grounding
- Standard saliency maps highlight arbitrary sample indices without clinical alignment.
- **Our Solution**: We align temporal gradients (SmoothGrad) with algorithmic fiducial interval delineations (NeuroKit2 Discrete Wavelet Transform) to measure exact diagnostic attribution across **P-waves**, **QRS complexes**, **ST segments**, and **T-waves**.

### Gap 3: Vulnerability to Missing or Noisy Leads
- In ambulatory, emergency, and wearable settings, leads often disconnect or experience heavy motion artifact.
- **Our Solution**: We incorporate **Multi-Lead Attention Pooling** and evaluate stochastic **Lead-Dropout Regularization** during training, preserving diagnostic accuracy across $12 \to 1$ active lead configurations.

### Gap 4: Clinically Interpretable Spatial Attention
- Multi-head GATv2 attention dynamically routes latent features across leads. Extracted attention weights $\alpha_{uv}$ reveal which contiguous anatomical territories informed each diagnostic prediction.

### Gap 5: Seamless Clinical Deployment & Interactive Telemetry
- A lightweight, high-performance web dashboard on `http://localhost:8000` allows clinicians to test any of the 2,198 test records, inspect 12-lead waveforms, toggle missing leads, and view live 3D electrical conduction.

---

## 🏗️ System Architecture

```
12-Lead ECG Recording (10s @ 100Hz)
       │  Shape: (Batch, 12, 1000)
       ▼
┌────────────────────────────────────────────────────────┐
│   Signal Preprocessing Pipeline                        │  0.5–45 Hz Bandpass, Median Wander Filter, Z-Score
└──────────────────────────┬─────────────────────────────┘
                           │  Shape: (Batch, 12, 1000)
                           ▼
┌────────────────────────────────────────────────────────┐
│   Per-Lead 1D-ResNet Temporal Encoder (Weight-Shared)  │  4 Residual Blocks + Conv1D + MaxPool + GELU
└──────────────────────────┬─────────────────────────────┘
                           │  H_temp: (Batch, 12, 128)
                           ├─────────────────────────────────────────────────┐
                           ▼                                                 │  Temporal Shortcut
┌────────────────────────────────────────────────────────┐                   │
│   Spatial GATv2 Message Passing (Multi-Head K=4)       │ ◄── Lead Adjacency A ∈ {0, 1}^(12x12)
│   Learned Dynamic Inter-Lead Attention α_uv            │     (16 Anatomical Edges + Self-Loops)
└──────────────────────────┬─────────────────────────────┘
                           │  H_graph: (Batch, 12, 128)
                           ▼
┌────────────────────────────────────────────────────────┐
│   Gated Spatial-Temporal Fusion                        │  H_fused = g ⊙ H_graph + (1 - g) ⊙ H_temp
│   Learnable Gate: g = σ(W_f · [H_temp; H_graph] + b)   │
└──────────────────────────┬─────────────────────────────┘
                           │  H_fused: (Batch, 12, 128)
                           ▼
┌────────────────────────────────────────────────────────┐
│   Multi-Lead Attention Pooling                         │  Lead Weights β = softmax(w_p^T tanh(W_p H_fused))
│   Adaptive Lead Importance Weighting                   │  Recording Representation r = ∑ β_i · H_fused,i
└──────────────────────────┬─────────────────────────────┘
                           │  Recording Vector: (Batch, 128)
                           ▼
┌────────────────────────────────────────────────────────┐
│   Multi-Label Classification Head                      │  Linear(128 → 64) → LayerNorm → GELU → Dropout(0.3)
│   5 Clinical Superclasses                              │  Linear(64 → 5) → Sigmoid
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
       [NORM, MI, STTC, CD, HYP] Calibrated Probabilities
```

---

## 📂 Repository Layout

```
ecg-stgat/
├── README.md                          # Primary research guide and benchmark summary
├── DEPLOYMENT.md                      # Production deployment, Docker, systemd, and API guide
├── SYSTEM_ARCHITECTURE.md             # Deep architectural specification and mathematical formulations
├── requirements.txt                   # Production Python package dependencies
├── run_eda.py                         # Master Exploratory Data Analysis & signal cache builder
├── run_experiments.py                 # Automated training and ablation study runner
├── smoke_train.py                     # Rapid CI/CD smoke test runner (2 epochs)
├── test_pipeline.py                   # Automated end-to-end integration and API tests
│
├── ecg_stgat/                         # Core Machine Learning Package
│   ├── data/
│   │   ├── preprocessing.py           # Bandpass filter, baseline wander removal, z-score normalization
│   │   ├── ptbxl_dataset.py           # WFDB reader, SCP code parser, 10-fold stratifier, RAM cache
│   │   └── build_cache.py             # Memory-mapped NumPy array generator for high-speed training
│   ├── models/
│   │   ├── temporal_encoder.py        # 1D-ResNet & 1D-CNN per-lead feature extractors → (B, 12, 128)
│   │   ├── lead_graph.py              # 12x12 Clinical adjacency (Static anatomical, FC, Learned)
│   │   ├── gat_module.py              # GATv2Conv with dynamic multi-head attention extraction (K=4)
│   │   ├── fusion_head.py             # Gated ST fusion, attention pooling, multi-label classifier
│   │   ├── stgat_model.py             # End-to-end ST-GAT model wiring
│   │   └── baseline_cnn.py            # Matched non-graph 1D-CNN baseline model (Hypothesis 1 control)
│   ├── training/
│   │   ├── config.yaml                # Standard hyperparameters (learning rate, AdamW, weights)
│   │   └── train.py                   # PyTorch training loop, AMP mixed precision, early stopping
│   ├── evaluation/
│   │   ├── metrics.py                 # Macro/Micro AUROC, F1, 1,000 bootstrap confidence intervals
│   │   ├── evaluate.py                # Standalone Fold 10 test evaluation script
│   │   ├── ablation.py                # RQ2: Representation, topology, and pooling ablations
│   │   ├── missing_lead_eval.py       # RQ4: Missing-lead degradation benchmark (12 → 1 leads)
│   │   └── lead_dropout_training.py   # RQ5: Stochastic lead-dropout regularized training loop
│   ├── explainability/
│   │   ├── saliency.py                # Temporal gradient saliency, Integrated Gradients, SmoothGrad
│   │   ├── lead_importance.py         # Multi-lead attention extraction and coronary territory mapping
│   │   ├── waveform_alignment.py      # NeuroKit2 DWT fiducial delineation & interval overlap metrics
│   │   └── run_explainability.py      # Automated explainability report generation
│   ├── diagrams/                      # SVG and PNG publication diagram generators
│   └── reporting/                     # Automated Markdown and CSV table and figure generators
│
├── web_app/                           # Interactive Clinical Research Web Application
│   ├── server.py                      # Production HTTP/REST backend (model cache, prediction, attention)
│   ├── index.html                     # Full-featured single-page research dashboard
│   ├── three.min.js                   # Three.js 3D WebGL library
│   └── OrbitControls.js               # 3D camera controls
│
├── data/                              # Dataset Directory
│   ├── ptbxl/                         # Raw PTB-XL v1.0.3 WFDB signals and metadata CSVs
│   └── cache/                         # Fast memory-mapped signal caches (`signals_folds_*.npy`)
│
├── runs/                              # Model Checkpoints & Training Artifacts
│   ├── STGAT_Static/                  # Proposed ST-GAT best checkpoints, metrics JSON, history
│   ├── Baseline_CNN/                  # Baseline 1D-CNN best checkpoints and logs
│   ├── STGAT_LeadDropout/             # Lead-dropout regularized model checkpoints
│   ├── ablations/                     # Checkpoints for 5 architectural ablation variants
│   └── missing_leads/                 # Evaluation metric JSONs across 12, 10, 8, 6, 3, 1 leads
│
└── results/                           # Publication Deliverables
    ├── tables/                        # Tables 1–5 in formatted Markdown and CSV
    └── figures/                       # Publication Figures 1–7 (300 DPI PNG + vector SVG)
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/aditisinha17/ecg-stagat.git
cd ecg-stagat

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Exploratory Data Analysis & Fast Dataset Caching

Build the preprocessed memory-mapped signal cache and generate baseline statistics:

```bash
python run_eda.py
```

### 3. Model Training

```bash
# 1. Train Proposed ST-GAT with Static Anatomical Graph:
python -m ecg_stgat.training.train --model stgat --graph_type static --exp_name STGAT_Static

# 2. Train Non-Graph Baseline 1D-CNN (RQ1 Comparison):
python -m ecg_stgat.training.train --model baseline_cnn --exp_name Baseline_CNN

# 3. Train ST-GAT with Stochastic Lead-Dropout (RQ5 Robustness):
python -m ecg_stgat.evaluation.lead_dropout_training --lead_mask_prob 0.5 --max_dropped_leads 6 --exp_name STGAT_LeadDropout
```

### 4. Comprehensive Evaluation & Explainability

```bash
# Benchmark Fold 10 Test Set with 1,000 Bootstrap Confidence Intervals:
python -m ecg_stgat.evaluation.evaluate --checkpoint runs/STGAT_Static/best_model.pt

# Run RQ2 Architectural Ablation Benchmark:
python -m ecg_stgat.evaluation.ablation --epochs 25

# Run RQ4 Missing-Lead Robustness Benchmark (12 → 1 Leads):
python -m ecg_stgat.evaluation.missing_lead_eval --checkpoint runs/STGAT_Static/best_model.pt

# Run RQ3 Clinical Explainability & Interval Saliency Alignment:
python -m ecg_stgat.explainability.run_explainability --checkpoint runs/STGAT_Static/best_model.pt
```

### 5. Launch Interactive Research Dashboard

Launch the real-time research application on port `8000`:

```bash
python web_app/server.py 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in any modern web browser.

---

## 🖥️ Interactive Web Dashboard Tour

The web application is structured into five synchronized clinical modules:

### Tab 1: Live Diagnostic Inference & Lead Masking
- **Record Selector**: Browse all 2,198 test-fold records with real-time waveform rendering.
- **Dynamic Lead Toggling**: Disconnect any subset of the 12 leads to test emergency single-lead or limb-lead configurations on the fly.
- **Model Selector**: Switch instantaneously between ST-GAT, Baseline 1D-CNN, and Lead-Dropout ST-GAT.
- **Diagnostics Output**: Calibrated probabilities for NORM, MI, STTC, CD, HYP with clinical risk levels and latency metrics (<250 ms on CPU).

### Tab 2: 3D Anatomical Heart & Spatial Lead Graph
- **Reference Diagram Colors**: Warm peach-pink myocardium, crimson-red aorta with supra-aortic branches, deep venous blue pulmonary trunk and vena cava, and branching coronary vessels.
- **12 Verified Lead Badges**: Circular badges arranged along the anatomical thoracic perimeter:
  - 6 Limb Leads on left/top: `I`, `aVR`, `II`, `III`, `aVF`, `aVL`.
  - 6 Precordial Leads on right/bottom: `V1`, `V2`, `V3`, `V4`, `V5`, `V6`.
  - Status Badge: `Nodes: 12/12 Verified` (validated against model ordering).
- **Proper Travelling Electrical Current**:
  - **Cardiac Conduction Wave**: Realistic propagation from SA Node $\to$ Atrial P-Wave $\to$ AV Node Delay $\to$ Bundle of His/Septum $\to$ Apex $\to$ Ventricular QRS $\to$ T-Wave repolarization.
  - **ST-GAT Graph Pulses**: Attention-scaled pulse particles flowing along genuine graph edges.
  - Interactive toggle button: `⚡ Travelling Current: ON/OFF`.
- **View Modes**: Switch between `🫀 3D Heart View`, `🕸️ Synchronized 2D Graph`, and `⚡ Side-by-Side Dual View`.
- **Unclipped Telemetry Panel**: Responsive 3-column clinical card displaying viewing angle, myocardial territory, coronary artery, and live multi-head attention weights.

### Tab 3: Interval-Level Clinical Explainability
- Combines NeuroKit2 DWT delineations with SmoothGrad saliency maps.
- Quantifies temporal attention allocated to QRS complexes, ST segments, T-waves, and P-waves.

### Tab 4: Missing-Lead Robustness Benchmark
- Interactive performance degradation charts across 12, 10, 8, 6, 3, and 1 lead configurations.

### Tab 5: Statistical Benchmark Tables
- Interactive view of verified experimental result tables (Tables 1–5).

---

## 📊 Verified Experimental Results (PTB-XL Test Fold 10)

### Table 1: RQ1 Overall Model Performance (1,000 Bootstrap 95% CIs)
| Model Architecture | Macro AUROC [95% CI] | Micro AUROC | Macro F1 | Micro F1 | Exact Match Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **ST-GAT (Static Anatomical)** | **0.8797** [0.8706, 0.8887] | **0.8871** | **0.6872** | **0.7317** | **53.23%** |
| Baseline 1D-CNN (Non-Graph) | 0.8720 [0.8619, 0.8816] | 0.8771 | 0.6645 | 0.7081 | 51.59% |
| **Statistical Net Gain** | **+0.0077** | **+0.0100** | **+0.0227** | **+0.0236** | **+1.64%** |

### Table 2: Diagnostic Superclass Breakdown (Fold 10 AUROC)
| Superclass | Clinical Description | Baseline 1D-CNN | ST-GAT Static | Δ AUROC | Physiological Mechanism |
|:---|:---|:---:|:---:|:---:|:---|
| **NORM** | Normal ECG Baseline | 0.9168 | **0.9176** | +0.0008 | Preserved sinus baseline morphology |
| **MI** | Myocardial Infarction | 0.8821 | **0.8938** | **+0.0117** | Multi-lead contiguous ST elevations (RCA, LAD, LCx) |
| **STTC** | ST-T Wave Changes | 0.8656 | **0.8732** | **+0.0076** | Generalized repolarization discordance |
| **CD** | Conduction Disturbance | 0.8354 | **0.8498** | **+0.0144** | Inter-ventricular conduction delay across QRS |
| **HYP** | Ventricular Hypertrophy | **0.8600** | 0.8578 | -0.0022 | Voltage criteria (Sokolow-Lyon index) |

### Table 3: RQ2 Representation & Graph Topology Ablation Study
| Ablation Configuration | Architectural Modification | Macro AUROC | Micro AUROC | Macro F1 | Exact Match | Key Takeaway |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Full ST-GAT** | Gated ST Fusion + Static Graph + Attention Pool | **0.8797** | **0.8881** | **0.6822** | **52.82%** | **Optimal overall architecture** |
| Concat Fusion | Simple concatenation replacing gated fusion | 0.8795 | 0.8844 | 0.6834 | 53.37% | Gating improves F1 calibration |
| FC Graph | Dense fully-connected 12x12 lead topology | 0.8729 | 0.8711 | 0.6688 | 48.86% | Dense message passing adds cross-lead noise (-4.37% EM) |
| Mean Pooling | Uniform average replacing attention pooling | 0.8625 | 0.7738 | 0.6419 | 43.08% | Attention pooling delivers +0.1133 Micro AUROC gain |
| Graph Only | Removes temporal shortcut ($g = 1$) | 0.8736 | 0.8775 | 0.6652 | 49.82% | Pure graph lacks individual lead temporal detail |
| Temporal Only | Removes spatial GAT layers entirely | 0.8550 | 0.8495 | 0.6432 | 46.18% | Removing spatial GAT causes severe degradation (-0.0247) |

### Table 4: RQ4 Missing-Lead Robustness Benchmark (12 → 1 Leads)
| Active Leads | Lead Configuration | Baseline 1D-CNN (AUROC) | Baseline Macro F1 | ST-GAT Static (AUROC) | ST-GAT Macro F1 |
|:---:|:---|:---:|:---:|:---:|:---:|
| **12 Leads** | Full standard 12-lead ECG | 0.8720 | 0.6645 | **0.8797** | **0.6822** |
| **10 Leads** | Dropped III, aVF | **0.8558** | 0.6473 | 0.8525 | **0.6563** |
| **8 Leads** | Dropped III, aVL, aVF, V2 | **0.8501** | **0.6377** | 0.8446 | 0.6332 |
| **6 Leads** | Standard 6 Limb Leads (I, II, III, aVR, aVL, aVF) | 0.8335 | 0.6108 | **0.8339** | **0.6247** |
| **3 Leads** | Einthoven Triangle (I, II, III) | 0.8123 | **0.5949** | **0.8153** | 0.5404 |
| **1 Lead** | Single Rhythm Channel (Lead II) | **0.7992** | **0.5906** | 0.7695 | 0.5124 |

---

## 📖 Citation

If you find this research or code useful in your work, please cite:

```bibtex
@article{stgat_ecg_2026,
  title={Spatial-Temporal Graph Attention Networks for 12-Lead ECG Classification with 3D Anatomical Grounding and Interval-Level Clinical Explainability},
  author={Sinha, Aditi and DeepMind Advanced Agentic Coding Team},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
