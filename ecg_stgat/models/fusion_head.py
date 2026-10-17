"""
Gated Spatial-Temporal Fusion, Multi-Lead Attention Pooling, and Multi-Label Head.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedSTFusion(nn.Module):
    """
    Learns dynamic gate g in [0, 1] to combine temporal features H_temp and graph features H_graph:
    H_fused = g * H_graph + (1 - g) * H_temp
    """
    def __init__(self, embed_dim: int, fusion_type: str = "gated"):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        if self.fusion_type == "gated":
            self.gate_proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
        elif self.fusion_type == "concat":
            self.proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, h_temp: torch.Tensor, h_graph: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "temporal_only":
            return h_temp
        if self.fusion_type == "graph_only":
            return h_graph
        if self.fusion_type == "concat":
            cat = torch.cat([h_temp, h_graph], dim=-1)
            return F.gelu(self.proj(cat))

        # Default gated
        cat = torch.cat([h_temp, h_graph], dim=-1)
        g = self.gate_proj(cat)
        return g * h_graph + (1.0 - g) * h_temp


class MultiLeadAttentionPooling(nn.Module):
    """
    Adaptively pools across 12 leads based on attention weights.
    Returns:
        recording_vec: (B, D)
        weights: (B, 12)
    """
    def __init__(self, embed_dim: int, pooling_type: str = "attention"):
        super().__init__()
        self.pooling_type = pooling_type.lower()
        if self.pooling_type == "attention":
            self.score_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.Tanh(),
                nn.Linear(embed_dim // 2, 1, bias=False)
            )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: (B, 12, D)
        """
        B, num_leads, D = h.shape
        if self.pooling_type == "mean":
            weights = torch.full((B, num_leads), 1.0 / num_leads, device=h.device, dtype=h.dtype)
            pooled = h.mean(dim=1)
            return pooled, weights

        scores = self.score_net(h).squeeze(-1)  # (B, 12)
        weights = F.softmax(scores, dim=-1)  # (B, 12)
        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, D)
        return pooled, weights


class ClassificationHead(nn.Module):
    """
    Multi-label classification head mapping recording vector to logits.
    """
    def __init__(self, embed_dim: int = 128, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
