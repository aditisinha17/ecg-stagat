"""
Verification script for Environment, Data Pipeline, and Model Architectures.
"""
import sys
import torch
import numpy as np

print("--- System & CUDA Check ---")
print("Python:", sys.version)
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from ecg_stgat.models.stgat_model import STGATModel
from ecg_stgat.models.baseline_cnn import BaselineCNNModel
from ecg_stgat.models.lead_graph import LeadGraphBuilder
from ecg_stgat.data.preprocessing import preprocess_ecg_signal

print("\n--- Testing Signal Preprocessing ---")
raw_sig = np.random.randn(12, 1000).astype(np.float32)
clean_sig = preprocess_ecg_signal(raw_sig, fs=100.0)
print(f"Preprocessed signal shape: {clean_sig.shape}, mean: {clean_sig.mean():.4f}, std: {clean_sig.std():.4f}")

print("\n--- Testing Lead Graph Builders ---")
static_gb = LeadGraphBuilder("static")
static_adj = static_gb()
print(f"Static Adjacency shape: {static_adj.shape}, non-zeros: {torch.count_nonzero(static_adj).item()}/144")

fc_gb = LeadGraphBuilder("fully_connected")
fc_adj = fc_gb()
print(f"FC Adjacency shape: {fc_adj.shape}, non-zeros: {torch.count_nonzero(fc_adj).item()}/144")

learned_gb = LeadGraphBuilder("learned")
learned_adj = learned_gb()
print(f"Learned Adjacency shape: {learned_adj.shape}, range: [{learned_adj.min().item():.3f}, {learned_adj.max().item():.3f}]")

print("\n--- Testing ST-GAT Model Forward & Backward ---")
batch_x = torch.randn(8, 12, 1000).to(device)
stgat = STGATModel(
    embed_dim=128,
    num_classes=5,
    num_heads=4,
    num_gat_layers=2,
    graph_type="static",
    fusion_type="gated",
    pooling_type="attention",
).to(device)

logits, exp = stgat(batch_x, return_explainability=True)
print(f"ST-GAT Output Logits Shape: {logits.shape}")
print(f"GAT Attention Weights Shape: {exp['gat_attention'].shape}")
print(f"Lead Pooling Weights Shape: {exp['lead_pooling_weights'].shape}")

# Test backward pass
loss = logits.sum()
loss.backward()
print("ST-GAT backward pass successful! Gradients populated.")

print("\n--- Testing Baseline CNN Model Forward & Backward ---")
baseline = BaselineCNNModel(embed_dim=128, num_classes=5).to(device)
b_logits, b_exp = baseline(batch_x, return_explainability=True)
print(f"Baseline Output Logits Shape: {b_logits.shape}")
print(f"Baseline Lead Weights Shape: {b_exp['lead_pooling_weights'].shape}")
b_loss = b_logits.sum()
b_loss.backward()
print("Baseline backward pass successful! Gradients populated.")

print("\n>>> ALL SMOKE TESTS PASSED PERFECTLY! <<<")
