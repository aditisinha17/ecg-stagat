# ECG-STGAT: Comprehensive Technical Reference & Developer Guide

This document provides the definitive, low-level engineering reference for developers, biomedical data scientists, and ML systems engineers working on or extending the **ECG-STGAT** (Spatial-Temporal Graph Attention Network) codebase.

---

## 📑 Table of Contents
1. [Engineering Stack & Software Dependencies](#1-engineering-stack--software-dependencies)
2. [End-to-End Tensor Lifecycle & Shape Transformations](#2-end-to-end-tensor-lifecycle--shape-transformations)
3. [Module-by-Module API Reference](#3-module-by-module-api-reference)
   - [3.1 ecg_stgat.data](#31-ecg_stgatdata)
   - [3.2 ecg_stgat.models](#32-ecg_stgatmodels)
   - [3.3 ecg_stgat.training](#33-ecg_stgattraining)
   - [3.4 ecg_stgat.evaluation](#34-ecg_stgatevaluation)
   - [3.5 ecg_stgat.explainability](#35-ecg_stgatexplainability)
4. [Configuration Reference (`config.yaml`)](#4-configuration-reference-configyaml)
5. [Frontend & WebGL 3D Visualization Internals](#5-frontend--webgl-3d-visualization-internals)
6. [Testing, Benchmarking & Extension Protocols](#6-testing-benchmarking--extension-protocols)

---

## 1. Engineering Stack & Software Dependencies

### Core ML & Numerical Runtime
- **Python**: `>= 3.10, < 3.13` (Tested extensively on Python 3.11.6 x86_64)
- **PyTorch**: `>= 2.4.0` (Supports both CUDA 11.8/12.1+ GPU acceleration and CPU SIMD AVX2/AVX-512)
- **NumPy**: `>= 1.26.0` (Zero-copy binary memory mapping via `numpy.lib.format.open_memmap`)
- **SciPy**: `>= 1.13.0` (Biomedical DSP routines: `scipy.signal.butter`, `filtfilt`, `medfilt`)
- **Pandas**: `>= 2.2.0` (PTB-XL metadata manipulation and SCP code parsing)
- **Scikit-Learn**: `>= 1.4.0` (Stratified split validation, ROC curves, multi-label F1 metrics)

### Biomedical Signal Analysis & Clinical NLP
- **WFDB (Waveform Database)**: `>= 4.1.2` (PhysioNet raw binary signal ingestion)
- **NeuroKit2**: `>= 0.2.7` (Algorithmic Discrete Wavelet Transform ECG wave delineation)

### Visualization & Web Serving
- **Three.js**: `r128` (High-performance WebGL 3D scene rendering)
- **OrbitControls.js**: Smooth 360-degree anatomical camera manipulation
- **Matplotlib & Seaborn**: Static publication figure generation (300 DPI PNG + vector SVG)

---

## 2. End-to-End Tensor Lifecycle & Shape Transformations

The following table traces the transformation of an ECG recording through every architectural stage of the pipeline:

| Pipeline Stage | Operation / Function | Input Tensor Shape | Output Tensor Shape | Data Type | Device |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Ingestion** | `WFDB` / `PTBXLDataset` | Disk `.dat` / `.hea` | `(12, 1000)` | `float32` | CPU |
| **0.5–45Hz Filter** | `butter_bandpass_filter` | `(12, 1000)` | `(12, 1000)` | `float64` | CPU |
| **Baseline Removal**| `remove_baseline_wander` | `(12, 1000)` | `(12, 1000)` | `float64` | CPU |
| **Standardization** | `normalize_signals` | `(12, 1000)` | `(12, 1000)` | `float32` | CPU |
| **Batch Collation** | `DataLoader` | List of `(12, 1000)` | `(B, 12, 1000)` | `float32` | CPU $\to$ GPU |
| **Temporal ResNet** | `TemporalResNetEncoder` | `(B, 12, 1000)` | `(B, 12, 128)` | `float32` | GPU / CPU |
| **Graph Adjacency** | `LeadGraphBuilder` | N/A | `(12, 12)` | `float32` | GPU / CPU |
| **GATv2 Layer 1** | `GATv2Layer` | `(B, 12, 128)` | `(B, 12, 128)` | `float32` | GPU / CPU |
| **GATv2 Layer 2** | `GATv2Layer` | `(B, 12, 128)` | `(B, 12, 128)` | `float32` | GPU / CPU |
| **Attention Tensor**| `GATv2Layer.forward` | N/A | `(B, 4, 12, 12)` | `float32` | GPU / CPU |
| **ST Gated Fusion** | `GatedSTFusion` | `(B, 12, 128)`, `(B, 12, 128)` | `(B, 12, 128)` | `float32` | GPU / CPU |
| **Attention Pooling**| `MultiLeadAttentionPooling`| `(B, 12, 128)` | `(B, 128)` | `float32` | GPU / CPU |
| **Lead Weights** | `MultiLeadAttentionPooling`| `(B, 12, 128)` | `(B, 12)` | `float32` | GPU / CPU |
| **Classification** | `ClassificationHead` | `(B, 128)` | `(B, 5)` | `float32` | GPU / CPU |
| **Inference Output**| `torch.sigmoid` | `(B, 5)` | `(B, 5)` | `float32` | GPU / CPU |

---

## 3. Module-by-Module API Reference

### 3.1 `ecg_stgat.data`

#### `preprocessing.py`
Provides deterministic, zero-phase signal conditioning:

- **`butter_bandpass_filter(signals: np.ndarray, lowcut: float = 0.5, highcut: float = 45.0, fs: float = 100.0, order: int = 3) -> np.ndarray`**
  - **Purpose**: Applies a forward-backward Butterworth bandpass filter across all channels.
  - **Inputs**: `signals` shape `(12, T)`.
  - **Returns**: Filtered NumPy array of shape `(12, T)` with zero phase distortion.

- **`remove_baseline_wander(signals: np.ndarray, fs: float = 100.0) -> np.ndarray`**
  - **Purpose**: Cascaded dual median filter ($200\text{ ms}$ QRS suppression followed by $600\text{ ms}$ T-wave suppression).
  - **Formula**: $\mathbf{X}_{\text{clean}} = \mathbf{X} - \text{Median}_{600}(\text{Median}_{200}(\mathbf{X}))$.

- **`normalize_signals(signals: np.ndarray, eps: float = 1e-6) -> np.ndarray`**
  - **Purpose**: Per-lead z-score standardization ensuring zero mean and unit variance.

- **`preprocess_ecg_recording(raw_signals: np.ndarray, fs: float = 100.0) -> np.ndarray`**
  - **Purpose**: Full sequential wrapper executing Bandpass $\to$ Wander Removal $\to$ Z-Score Normalization.

#### `ptbxl_dataset.py`
Handles PhysioNet PTB-XL v1.0.3 data structures:

- **`load_raw_ptbxl_metadata(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]`**
  - Reads `ptbxl_database.csv` and `scp_statements.csv`. Returns metadata DataFrame and diagnostic statements dictionary.

- **`map_scp_to_superclasses(scp_dict: dict, scp_statements: pd.DataFrame) -> List[str]`**
  - Maps 71 granular clinical SCP statements into the 5 official PhysioNet superclasses: `NORM`, `MI`, `STTC`, `CD`, `HYP`.

- **`class PTBXLDataset(torch.utils.data.Dataset)`**
  - High-throughput PyTorch Dataset.
  - **Constructor**: `PTBXLDataset(df, signals_cache, transform=None)`
  - **`__getitem__(idx)`**: Returns `(signals_tensor: torch.FloatTensor (12, 1000), labels_tensor: torch.FloatTensor (5), ecg_id: int)`.

---

### 3.2 `ecg_stgat.models`

#### `temporal_encoder.py`

- **`class ResNetBlock1D(nn.Module)`**
  - **Parameters**: `in_channels: int`, `out_channels: int`, `stride: int = 1`.
  - **Structure**: Conv1D(kernel=7, pad=3) $\to$ BatchNorm1d $\to$ GELU $\to$ Conv1D(kernel=7, pad=3) $\to$ BatchNorm1d + Shortcut $\to$ GELU.

- **`class TemporalResNetEncoder(nn.Module)`**
  - **Parameters**: `in_channels: int = 1`, `embed_dim: int = 128`.
  - **Input**: Tensor of shape `(B, 12, T)`.
  - **Operation**: Flattens leads into batch dimension `(B * 12, 1, T)`, processes through 4 residual stages with stride-2 downsampling, performs `AdaptiveAvgPool1d(1)`, and projects through a linear layer.
  - **Output**: `(B, 12, embed_dim)`.

#### `lead_graph.py`

- **`build_static_anatomical_adjacency() -> torch.Tensor`**
  - Returns a binary symmetric tensor `(12, 12)` encoding the 16 clinical edges + 12 self-loops.
  - Non-zero count: $44 / 144$ entries (sparsity $\approx 69.4\%$).

- **`build_fully_connected_adjacency() -> torch.Tensor`**
  - Returns an all-ones tensor `(12, 12)` used in RQ2 ablation studies.

- **`class LearnedAdjacencyMatrix(nn.Module)`**
  - Learnable parameter matrix $\mathbf{\Theta} \in \mathbb{R}^{12 \times 12}$ passed through a sigmoid function $\mathbf{A} = \sigma(\mathbf{\Theta})$.

- **`class LeadGraphBuilder(nn.Module)`**
  - Factory module dispatching `"static"`, `"fully_connected"`, or `"learned"` graphs.

#### `gat_module.py`

- **`class GATv2Layer(nn.Module)`**
  - Implements dynamic multi-head GATv2 attention (Brody et al., 2022).
  - **Attention Score Equation**:
    $$e_{ij}^k = \mathbf{a}_k^T \, \text{LeakyReLU}\left(\mathbf{W}_L^k \mathbf{h}_i + \mathbf{W}_R^k \mathbf{h}_j\right)$$
  - **Graph Masking**: Disconnected pairs ($\mathbf{A}_{ij} \le 10^{-6}$) are filled with $-10^9$ prior to softmax:
    $$\alpha_{ij}^k = \frac{\exp(e_{ij}^k)}{\sum_{l \in \mathcal{N}(i)} \exp(e_{il}^k)}$$
  - **Message Passing**: $\mathbf{h}_i' = \text{LayerNorm}\left(\mathbf{h}_i + \mathbf{W}_O \left[\prod_{k=1}^K \sum_{j} \alpha_{ij}^k \mathbf{W}_V^k \mathbf{h}_j\right]\right)$.
  - **Returns**: `(out: Tensor (B, 12, D), attention_weights: Tensor (B, K, 12, 12))`.

- **`class GATv2Module(nn.Module)`**
  - Chains $N$ stacked `GATv2Layer` instances (default: 2 layers, $K = 4$ heads).

#### `fusion_head.py`

- **`class GatedSTFusion(nn.Module)`**
  - **Formula**:
    $$\mathbf{g}_i = \sigma(\mathbf{W}_g [\mathbf{h}_i^{\text{temp}} \,\|\, \mathbf{h}_i^{\text{graph}}] + \mathbf{b}_g)$$
    $$\mathbf{h}_i^{\text{fused}} = \mathbf{g}_i \odot \mathbf{h}_i^{\text{graph}} + (1 - \mathbf{g}_i) \odot \mathbf{h}_i^{\text{temp}}$$
  - Supports ablation modes: `"gated"`, `"concat"`, `"temporal_only"`, `"graph_only"`.

- **`class MultiLeadAttentionPooling(nn.Module)`**
  - Computes context-dependent lead weights $\beta_i \in [0, 1]$ such that $\sum_{i=1}^{12} \beta_i = 1$:
    $$u_i = \mathbf{v}^T \tanh(\mathbf{W}_p \mathbf{h}_i^{\text{fused}} + \mathbf{b}_p), \quad \beta = \text{softmax}(\mathbf{u})$$
    $$\mathbf{r} = \sum_{i=1}^{12} \beta_i \mathbf{h}_i^{\text{fused}} \in \mathbb{R}^{D}$$
  - Returns `(recording_vector: (B, D), lead_weights: (B, 12))`.

- **`class ClassificationHead(nn.Module)`**
  - Linear($D \to D$) $\to$ LayerNorm $\to$ GELU $\to$ Dropout(0.2) $\to$ Linear($D \to 5$).

#### `stgat_model.py`

- **`class STGATModel(nn.Module)`**
  - Master model combining `TemporalResNetEncoder`, `LeadGraphBuilder`, `GATv2Module`, `GatedSTFusion`, `MultiLeadAttentionPooling`, and `ClassificationHead`.
  - **`forward(x, return_explainability=False)`**:
    - If `return_explainability=False`: Returns raw logits `(B, 5)`.
    - If `return_explainability=True`: Returns `(logits, explainability_dict)` containing:
      - `"gat_attention"`: Tensor `(B, 4, 12, 12)`
      - `"lead_pooling_weights"`: Tensor `(B, 12)`
      - `"h_temp"`: Tensor `(B, 12, 128)`
      - `"h_fused"`: Tensor `(B, 12, 128)`
      - `"rec_vec"`: Tensor `(B, 128)`

---

### 3.3 `ecg_stgat.training`

#### `train.py`
Production PyTorch training pipeline:
- **Optimizer**: AdamW with decoupled weight decay ($\lambda = 10^{-4}$).
- **Learning Rate Scheduler**: Cosine Annealing with Warm Restarts or Plateau Decay.
- **Mixed Precision**: Automatic Mixed Precision (`torch.cuda.amp.autocast`) for high-throughput GPU execution.
- **Positional Class Weighting**: Automatically balances BCE loss according to inverse frequency ratios of the 5 superclasses.
- **Early Stopping**: Monitored on Validation Fold 9 Macro AUROC with a patience of 8 epochs.

---

### 3.4 `ecg_stgat.evaluation`

#### `metrics.py`
Provides clinically accepted statistical evaluation routines:
- **`compute_multilabel_metrics(y_true, y_pred_probs, thresholds=None) -> Dict`**:
  - Macro AUROC, Micro AUROC.
  - Per-class AUROC for `NORM`, `MI`, `STTC`, `CD`, `HYP`.
  - Macro F1, Micro F1, Exact Match Ratio (Subset Accuracy).
  - Average Precision (PR-AUC).
- **`find_optimal_thresholds(y_true, y_pred_probs) -> np.ndarray`**:
  - Maximizes Youden's J statistic or F1 score independently for each class.
- **`bootstrap_confidence_intervals(y_true, y_pred_probs, n_bootstraps=1000, alpha=0.95) -> Dict`**:
  - Computes non-parametric empirical 95% confidence intervals via 1,000 resamples with replacement.

---

### 3.5 `ecg_stgat.explainability`

#### `saliency.py`
- **`compute_smoothgrad(model, x, target_class_idx, n_samples=25, stdev_spread=0.15) -> np.ndarray`**:
  - Injects Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ into inputs $x$, averages input gradients $\frac{\partial \hat{y}_c}{\partial x}$, and returns a smooth temporal attribution map `(12, 1000)`.

#### `waveform_alignment.py`
- **`delineate_ecg_intervals_dwt(signal_1d, fs=100.0) -> Dict[str, List[Tuple[int, int]]]`**:
  - Uses Discrete Wavelet Transforms to detect R-peaks, Q-wave onsets, J-points, and T-wave offsets.
- **`compute_interval_overlap_attribution(saliency_map, intervals) -> Dict[str, float]`**:
  - Quantifies the proportion of gradient energy concentrated inside clinically defined P, QRS, ST, and T intervals.

---

## 4. Configuration Reference (`config.yaml`)

The training and evaluation pipeline is driven by `ecg_stgat/training/config.yaml`:

```yaml
# Data & Paths
data:
  dataset_name: "ptbxl"
  data_dir: "data/ptbxl"
  cache_dir: "data/cache"
  sampling_rate: 100           # Samples per second (T = 1000)
  num_classes: 5
  train_folds: [1, 2, 3, 4, 5, 6, 7, 8]
  val_folds: [9]
  test_folds: [10]

# Model Architecture
model:
  embed_dim: 128               # Dimension of lead embeddings (D)
  temporal_encoder: "resnet1d" # 'resnet1d' or 'cnn1d'
  graph_type: "static"         # 'static', 'fully_connected', or 'learned'
  num_heads: 4                 # GATv2 attention heads (K)
  num_gat_layers: 2            # Number of stacked GATv2 message passing steps
  fusion_type: "gated"         # 'gated', 'concat', 'temporal_only', 'graph_only'
  pooling_type: "attention"    # 'attention' or 'mean'
  dropout: 0.20                # Classifier dropout probability

# Training Hyperparameters
training:
  batch_size: 64
  epochs: 50
  learning_rate: 0.00075
  weight_decay: 0.0001
  amp: true                    # Enable PyTorch Automatic Mixed Precision
  early_stopping_patience: 8
  loss_weights:
    NORM: 1.0
    MI: 1.73
    STTC: 1.81
    CD: 1.94
    HYP: 3.57
```

---

## 5. Frontend & WebGL 3D Visualization Internals

The interactive web dashboard (`web_app/index.html`) operates without complex build tools, combining vanilla ES6+ with Three.js r128:

```
                          [Three.js Scene Graph]
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
    [heartMeshGroup]         [leadNodesGroup]         [graphEdgesGroup]
    - Ventricles (Peach)     - 12 Circular Badges     - 16 Direct Cylinders
    - Atria & Auricles       - 12 Core Electrodes     - Live Attention Scaler
    - Crimson Aorta & Arch   - Pointer Raycaster      - Dynamic Edge Filter
    - Blue Pulmonary & SVC
    - Red/Blue Coronaries
           │
           ▼
 [travellingCurrentGroup]
 ├── Conduction Highway (SA Node → Atria → AV Crux → Bundle of His → Septum → Apex)
 └── ST-GAT Edge Pulses (16 pairs travelling at speeds proportional to attention α_uv)
```

### 1. Mathematical Lead Placement (Reference Diagram Layout)
The 12 electrodes are placed on an ellipse with radii $R_x = 3.20, R_y = 2.75$ surrounding the heart mesh without obscuring myocardium:
- **Limb Leads (Left & Top)**:
  - Lead I: `[0.00, 2.75, 0.00]` (Cyan `#06b6d4`, 12 o'clock)
  - Lead aVR: `[-2.20, 2.15, 0.00]` (Amber `#f59e0b`, 10:30 o'clock)
  - Lead II: `[-3.15, 0.80, 0.00]` (Emerald `#10b981`, 9:30 o'clock)
  - Lead III: `[-3.15, -0.80, 0.00]` (Teal `#14b8a6`, 8:30 o'clock)
  - Lead aVF: `[-2.20, -2.15, 0.00]` (Green `#22c55e`, 7:30 o'clock)
  - Lead aVL: `[2.20, 2.15, 0.00]` (Gold `#eab308`, 1:30 o'clock)
- **Precordial Leads (Right & Bottom)**:
  - Lead V1: `[3.15, 1.25, 0.00]` (Purple `#8b5cf6`, 2:30 o'clock)
  - Lead V2: `[3.35, 0.20, 0.00]` (Magenta `#ec4899`, 3:15 o'clock)
  - Lead V3: `[3.15, -0.85, 0.00]` (Coral `#f43f5e`, 4:00 o'clock)
  - Lead V4: `[2.45, -1.85, 0.00]` (Orange `#f97316`, 4:45 o'clock)
  - Lead V5: `[1.30, -2.60, 0.00]` (Violet `#6366f1`, 5:30 o'clock)
  - Lead V6: `[-0.20, -2.75, 0.00]` (Blue `#3b82f6`, 6:15 o'clock)

### 2. High-DPI Circular Badge Sprite Generator
Each badge is generated dynamically via an off-screen HTML canvas:
```javascript
const c = document.createElement("canvas");
c.width = 240; c.height = 240;
const ctx = c.getContext("2d");
// Glowing colored border ring (10px width)
ctx.arc(120, 120, 102, 0, Math.PI * 2);
ctx.fillStyle = "rgba(10, 15, 29, 0.94)";
ctx.fill();
ctx.lineWidth = 10;
ctx.strokeStyle = hexColor;
ctx.stroke();
// Bold white lead abbreviation
ctx.fillStyle = "#FFFFFF";
ctx.font = "bold 64px 'JetBrains Mono', Inter, monospace";
ctx.fillText(badgeText, 120, 120);
```

### 3. Travelling Current Conduction Math
- **Cardiac Highway Wavefront**: Follows a 7-point Catmull-Rom spline with cycle duration $T = 1.0\text{s}$ ($60\text{ bpm}$):
  $$\vec{P}(t) = \text{Spline}(\tau), \quad \tau = (t \bmod 1.0)$$
- **ST-GAT Graph Pulses**: For each edge $(u, v)$, particles move along:
  $$\vec{p}_{\text{fwd}}(t) = \vec{p}_u + (\vec{p}_v - \vec{p}_u) \cdot \left((t \cdot v_{\text{att}} + \phi) \bmod 1.0\right)$$
  Where $v_{\text{att}} = 0.65 + 0.85 \cdot \text{clamp}(3.5 \cdot (\alpha_{uv} + \alpha_{vu}), 0.15, 1.0)$.

---

## 6. Testing, Benchmarking & Extension Protocols

### Automated Regression & Smoke Testing
Run the complete test suite to verify model shapes, backward pass gradient flow, and preprocessing integrity:
```bash
python test_pipeline.py
```

### Adding a New Graph Topology
To test a custom graph structure (e.g., correlation-derived or metric-distance graph):
1. Open `ecg_stgat/models/lead_graph.py`.
2. Add a new builder function:
   ```python
   def build_custom_adjacency(...) -> torch.Tensor:
       # Return (12, 12) symmetric tensor
   ```
3. Register the new type in `LeadGraphBuilder.__init__`.
4. Run ablation comparison:
   ```bash
   python -m ecg_stgat.evaluation.ablation --graph_type custom
   ```

### Exporting to ONNX for Production Inference
```python
import torch
from ecg_stgat.models.stgat_model import STGATModel

model = STGATModel()
model.eval()
dummy_input = torch.randn(1, 12, 1000)

torch.onnx.export(
    model,
    dummy_input,
    "ecg_stgat.onnx",
    input_names=["ecg_signals"],
    output_names=["diagnostic_logits"],
    dynamic_axes={"ecg_signals": {0: "batch_size"}, "diagnostic_logits": {0: "batch_size"}},
    opset_version=17
)
print("Model exported to ecg_stgat.onnx successfully!")
```
