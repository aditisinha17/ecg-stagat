"""
Multi-Head Dynamic Graph Attention Network v2 (GATv2) Module.
Performs spatial message passing across 12 ECG leads with attention extraction.
"""
from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """
    Single GATv2 Layer supporting dynamic multi-head attention and adjacency masking.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.lin_src = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_dst = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att = nn.Parameter(torch.FloatTensor(num_heads, self.head_dim))
        self.lin_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.kaiming_uniform_(self.att, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, N, D) where N=12
            adj: (N, N) or (B, N, N) adjacency matrix
        Returns:
            out: (B, N, D)
            alpha: (B, num_heads, N, N) attention weights
        """
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim

        # Project and reshape to (B, H, N, d)
        src = self.lin_src(x).view(B, N, H, d).transpose(1, 2)
        dst = self.lin_dst(x).view(B, N, H, d).transpose(1, 2)
        v = self.lin_v(x).view(B, N, H, d).transpose(1, 2)

        # Compute pair-wise sum: (B, H, N, 1, d) + (B, H, 1, N, d) -> (B, H, N, N, d)
        scores = src.unsqueeze(3) + dst.unsqueeze(2)
        scores = self.leaky_relu(scores)

        # Inner product with attention vector: (B, H, N, N)
        # self.att: (H, d) -> unsqueeze to (1, H, 1, 1, d)
        att_vec = self.att.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        scores = (scores * att_vec).sum(dim=-1)

        # Apply graph structure mask
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).unsqueeze(1)  # (1, 1, N, N)
        elif adj.dim() == 3:
            adj = adj.unsqueeze(1)  # (B, 1, N, N)

        # Mask disconnected edges with -1e9
        mask = (adj <= 1e-6)
        scores = scores.masked_fill(mask, -1e9)

        alpha = F.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha_drop = self.dropout(alpha)

        # Message passing: (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
        out = torch.matmul(alpha_drop, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        out = self.norm(x + out)

        return out, alpha


class GATv2Module(nn.Module):
    """
    Stacked 2-layer GATv2 module for spatial message passing over 12 ECG leads.
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GATv2Layer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        last_att = None
        for layer in self.layers:
            h, last_att = layer(h, adj, return_attention=return_attention)
        return h, last_att
