"""
Full Spatial-Temporal Graph Attention Network (ST-GAT) Model Wiring.
Combines 1D-ResNet temporal encoder + GATv2 spatial graph attention + Gated Fusion + Attention Pooling.
"""
from typing import Dict, Tuple, Union, Optional
import torch
import torch.nn as nn

from ecg_stgat.models.temporal_encoder import TemporalResNetEncoder
from ecg_stgat.models.lead_graph import LeadGraphBuilder
from ecg_stgat.models.gat_module import GATv2Module
from ecg_stgat.models.fusion_head import GatedSTFusion, MultiLeadAttentionPooling, ClassificationHead


class STGATModel(nn.Module):
    """
    ST-GAT: Multi-Task Spatial-Temporal Graph Attention Network for 12-Lead ECG Classification.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 5,
        num_heads: int = 4,
        num_gat_layers: int = 2,
        graph_type: str = "static",
        fusion_type: str = "gated",
        pooling_type: str = "attention",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 1. Temporal feature extractor
        self.temporal_encoder = TemporalResNetEncoder(in_channels=1, embed_dim=embed_dim)

        # 2. Graph topology builder
        self.graph_builder = LeadGraphBuilder(graph_type=graph_type)

        # 3. Spatial message passing
        self.gat = GATv2Module(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_gat_layers,
            dropout=dropout
        )

        # 4. Fusion
        self.fusion = GatedSTFusion(embed_dim=embed_dim, fusion_type=fusion_type)

        # 5. Multi-lead pooling
        self.pooling = MultiLeadAttentionPooling(embed_dim=embed_dim, pooling_type=pooling_type)

        # 6. Classifier head
        self.classifier = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_explainability: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            x: (B, 12, T) raw or preprocessed 12-lead ECG signals
            return_explainability: if True, returns (logits, dict of explainability tensors)
        Returns:
            logits: (B, num_classes)
            exp: Dict with keys 'gat_attention', 'lead_pooling_weights', 'h_temp', 'h_fused'
        """
        # Temporal representations: (B, 12, D)
        h_temp = self.temporal_encoder(x)

        # Graph adjacency: (12, 12)
        adj = self.graph_builder()

        # Spatial message passing: (B, 12, D)
        h_graph, gat_att = self.gat(h_temp, adj, return_attention=return_explainability)

        # Gated ST fusion: (B, 12, D)
        h_fused = self.fusion(h_temp, h_graph)

        # Multi-lead attention pooling: (B, D), weights: (B, 12)
        rec_vec, lead_weights = self.pooling(h_fused)

        # Logits: (B, 5)
        logits = self.classifier(rec_vec)

        if return_explainability:
            exp = {
                "gat_attention": gat_att if gat_att is not None else torch.zeros(1, device=x.device),
                "lead_pooling_weights": lead_weights,
                "h_temp": h_temp,
                "h_fused": h_fused,
                "rec_vec": rec_vec,
            }
            return logits, exp

        return logits
