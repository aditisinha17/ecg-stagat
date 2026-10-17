| Leads | Configuration | Baseline 1D-CNN (AUROC) | Baseline Macro F1 | ST-GAT Static (AUROC) | ST-GAT Static F1 | ST-GAT Static Exact Match | ST-GAT Lead-Dropout (AUROC) | ST-GAT Lead-Dropout F1 | ST-GAT Lead-Dropout Exact Match |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **12L** | Standard 12-lead (Full) | 0.8720 | 0.6645 | **0.8797** | **0.6822** | **52.82%** | 0.8599 | 0.6573 | 46.09% |
| **10L** | Dropped III, aVF | **0.8558** | 0.6473 | 0.8525 | **0.6563** | **50.45%** | 0.8525 | 0.6512 | 45.50% |
| **8L** | Dropped III, aVL, aVF, V2 | 0.8501 | 0.6377 | 0.8446 | 0.6332 | 44.86% | **0.8569** | **0.6561** | **49.91%** |
| **6L** | 6 Limb Leads (I, II, III, aVR, aVL, aVF) | 0.8335 | 0.6108 | 0.8339 | **0.6247** | 40.22% | **0.8413** | 0.6204 | **43.95%** |
| **3L** | Einthoven Triangle (I, II, III) | 0.8123 | 0.5949 | 0.8153 | 0.5404 | 3.32% | **0.8358** | **0.6161** | **41.63%** (+12.5x) |
| **1L** | Single Rhythm Channel (Lead II) | **0.7992** | 0.5906 | 0.7695 | 0.5124 | 1.91% | 0.7943 | 0.5770 | **42.77%** (+22.4x) |

> [!IMPORTANT]
> **Key Finding (Empirical Resolution of Research Gap 3):**
> 1. **Covariate Shift in Static Graph**: When trained exclusively on complete 12-lead ECGs, `ST-GAT Static` suffers severe exact-match collapse when reduced to 3 leads (3.32%) or 1 lead (1.91%), demonstrating that unregularized spatial attention over-relies on full topological completeness.
> 2. **Lead-Dropout Resilience**: Incorporating stochastic Lead-Dropout during training regularizes the attention weights, preserving **41.63%** Exact Match at 3 leads (**+12.5x gain**) and **42.77%** Exact Match at 1 lead (**+22.4x gain**), while boosting 3-lead Macro AUROC from 0.8153 to **0.8358** (+0.0205) and Macro F1 from 0.5404 to **0.6161** (+0.0757).
