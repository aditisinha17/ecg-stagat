| Ablation Condition          | Component Modified                                        |   Macro AUROC |   Micro AUROC |   Macro F1 |   Micro F1 |   Exact Match Acc |
|:----------------------------|:----------------------------------------------------------|--------------:|--------------:|-----------:|-----------:|------------------:|
| Baseline 1D-CNN (Reference) | No spatial graph (temporal encoder only)                  |        0.8720 |        0.8771 |     0.6645 |     0.7081 |            0.5159 |
| Temporal Only               | Temporal encoder only (bypasses spatial graph)            |        0.8550 |        0.8495 |     0.6432 |     0.6901 |            0.4618 |
| Graph Only                  | Graph attention only (bypasses temporal shortcut)         |        0.8736 |        0.8775 |     0.6652 |     0.7120 |            0.4982 |
| Mean Pooling                | Uniform mean pooling across leads (replaces attention)    |        0.8625 |        0.7738 |     0.6419 |     0.6861 |            0.4308 |
| Concat Fusion               | Concatenation fusion instead of gated fusion              |        0.8795 |        0.8844 |     0.6834 |     0.7300 |            0.5337 |
| Fc Graph                    | Fully-connected dense graph (all-to-all control topology) |        0.8729 |        0.8711 |     0.6688 |     0.7127 |            0.4886 |
| Full ST-GAT (Proposed)      | Gated fusion + Static Lead Graph + Attention Pooling      |        0.8797 |        0.8881 |     0.6822 |     0.7282 |            0.5282 |