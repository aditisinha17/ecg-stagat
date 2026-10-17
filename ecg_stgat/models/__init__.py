"""
Model Architectures for ECG ST-GAT.
"""
from ecg_stgat.models.stgat_model import STGATModel
from ecg_stgat.models.baseline_cnn import BaselineCNNModel
from ecg_stgat.models.lead_graph import LeadGraphBuilder, build_static_anatomical_adjacency
from ecg_stgat.models.temporal_encoder import TemporalResNetEncoder
from ecg_stgat.models.gat_module import GATv2Module
from ecg_stgat.models.fusion_head import GatedSTFusion, MultiLeadAttentionPooling, ClassificationHead

__all__ = [
    "STGATModel",
    "BaselineCNNModel",
    "LeadGraphBuilder",
    "build_static_anatomical_adjacency",
    "TemporalResNetEncoder",
    "GATv2Module",
    "GatedSTFusion",
    "MultiLeadAttentionPooling",
    "ClassificationHead",
]
