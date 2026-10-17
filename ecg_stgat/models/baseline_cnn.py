"""
Baseline 1D-CNN Model (Non-Graph Comparison Baseline for RQ1 / H1).
Shares identical temporal encoder backbone and multi-lead attention pooling / head,
but excludes spatial graph message passing.
"""
from typing import Dict, Tuple, Union
import torch
import torch.nn as nn

from ecg_stgat.models.temporal_encoder import TemporalResNetEncoder
from ecg_stgat.models.fusion_head import MultiLeadAttentionPooling, ClassificationHead


class BaselineCNNModel(nn.Module):
    """
    Non-graph Baseline 1D-CNN Model for 12-lead ECG.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Shared 1D-ResNet temporal encoder
        self.temporal_encoder = TemporalResNetEncoder(in_channels=1, embed_dim=embed_dim)

        # Multi-lead attention pooling
        self.pooling = MultiLeadAttentionPooling(embed_dim=embed_dim, pooling_type="attention")

        # Classification head
        self.classifier = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_explainability: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Temporal representations: (B, 12, D)
        h_temp = self.temporal_encoder(x)

        # Multi-lead attention pooling: (B, D)
        rec_vec, lead_weights = self.pooling(h_temp)

        # Logits: (B, 5)
        logits = self.classifier(rec_vec)

        if return_explainability:
            exp = {
                "lead_pooling_weights": lead_weights,
                "h_temp": h_temp,
                "rec_vec": rec_vec,
            }
            return logits, exp

        return logits
