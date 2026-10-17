"""
Spatial GAT Attention Extraction and Lead Importance Profiling.
Extracts which leads the model prioritizes for each diagnostic superclass.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ecg_stgat.data.ptbxl_dataset import LEAD_NAMES, DIAGNOSTIC_CLASSES


def extract_lead_importance(
    model: nn.Module,
    ecg_tensor: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, np.ndarray]:
    """
    Extracts spatial attention weights and lead importance for a batch or single ECG.
    """
    model.eval()
    if ecg_tensor.dim() == 2:
        ecg_tensor = ecg_tensor.unsqueeze(0)

    with torch.no_grad():
        logits, exp = model(ecg_tensor.to(device), return_explainability=True)
        probs = torch.sigmoid(logits).cpu().numpy()
        lead_weights = exp["lead_pooling_weights"].cpu().numpy()
        gat_att = exp["gat_attention"].cpu().numpy()

    return {
        "probabilities": probs,
        "lead_weights": lead_weights,
        "gat_attention": gat_att,
        "lead_names": LEAD_NAMES,
    }
