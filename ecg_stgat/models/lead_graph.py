"""
12-Lead ECG Spatial Graph Topology Builder.
Implements:
1. Static Anatomical Lead Graph (Einthoven limb triangles, Goldberger augmented leads, precordial heart views)
2. Fully-Connected (FC) Lead Graph
3. Learned Dynamic Graph Topology
"""
import torch
import torch.nn as nn
from ecg_stgat.data.ptbxl_dataset import LEAD_NAMES


def build_static_anatomical_adjacency() -> torch.Tensor:
    """
    Constructs a 12x12 physiologically grounded static anatomical ECG lead adjacency matrix.
    Leads: ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    """
    num_leads = 12
    A = torch.zeros((num_leads, num_leads), dtype=torch.float32)

    # 1. Self-loops
    for i in range(num_leads):
        A[i, i] = 1.0

    # 2. Limb lead views (Einthoven triangle & Goldberger augmented)
    # Inferior group: II (1), III (2), aVF (5)
    inferior = [1, 2, 5]
    for i in inferior:
        for j in inferior:
            A[i, j] = 1.0

    # High Lateral limb group: I (0), aVL (4)
    lateral_limb = [0, 4]
    for i in lateral_limb:
        for j in lateral_limb:
            A[i, j] = 1.0

    # aVR (3) reciprocal connects to I (0), II (1), aVL (4)
    avr_conn = [0, 1, 4]
    for j in avr_conn:
        A[3, j] = 1.0
        A[j, 3] = 1.0

    # 3. Precordial anatomical views:
    # Septal: V1 (6), V2 (7)
    # Anterior: V3 (8), V4 (9)
    # Lateral precordial: V5 (10), V6 (11)
    precordial_groups = [[6, 7], [8, 9], [10, 11]]
    for grp in precordial_groups:
        for i in grp:
            for j in grp:
                A[i, j] = 1.0

    # Contiguous precordial sequence (V1-V2, V2-V3, V3-V4, V4-V5, V5-V6)
    for i in range(6, 11):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0

    # Lateral cross-connection (I, aVL with V5, V6)
    for limb in [0, 4]:
        for prec in [10, 11]:
            A[limb, prec] = 1.0
            A[prec, limb] = 1.0

    return A


class LeadGraphBuilder(nn.Module):
    """
    Generates adjacency matrix for 12-lead ECG graph convolution.
    Supported types: 'static', 'fully_connected', 'learned'
    """
    def __init__(self, graph_type: str = "static"):
        super().__init__()
        self.graph_type = graph_type.lower()
        if self.graph_type == "static":
            self.register_buffer("adj", build_static_anatomical_adjacency())
        elif self.graph_type == "fully_connected":
            self.register_buffer("adj", torch.ones((12, 12), dtype=torch.float32))
        elif self.graph_type == "learned":
            # Learnable adjacency initialized near static prior
            base_adj = build_static_anatomical_adjacency()
            self.raw_adj = nn.Parameter(base_adj + 0.1 * torch.randn(12, 12))
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")

    def forward(self) -> torch.Tensor:
        if self.graph_type in ("static", "fully_connected"):
            return self.adj
        elif self.graph_type == "learned":
            return torch.sigmoid(self.raw_adj)
        return self.adj
