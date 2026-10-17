"""
Gradient-based Temporal Saliency and SmoothGrad for 12-Lead ECG.
Identifies which temporal time steps and waveform intervals drive class predictions.
"""
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn


def compute_gradient_saliency(
    model: nn.Module,
    ecg_tensor: torch.Tensor,
    target_class_idx: int,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Computes Vanilla Gradient Saliency map for a single ECG sample.
    Args:
        ecg_tensor: Tensor of shape (1, 12, T) or (12, T)
        target_class_idx: Int in [0, 4]
    Returns:
        Saliency map of shape (12, T)
    """
    model.eval()
    if ecg_tensor.dim() == 2:
        ecg_tensor = ecg_tensor.unsqueeze(0)

    x = ecg_tensor.clone().to(device).requires_grad_(True)
    logits = model(x)
    target_logit = logits[0, target_class_idx]
    model.zero_grad()
    target_logit.backward()

    grads = x.grad.detach().cpu().squeeze(0).numpy()
    saliency = np.abs(grads)
    return saliency


def compute_smoothgrad(
    model: nn.Module,
    ecg_tensor: torch.Tensor,
    target_class_idx: int,
    num_samples: int = 20,
    stdev_spread: float = 0.15,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Computes SmoothGrad saliency by averaging gradients across noisy inputs.
    """
    model.eval()
    if ecg_tensor.dim() == 2:
        ecg_tensor = ecg_tensor.unsqueeze(0)

    x_np = ecg_tensor.squeeze(0).cpu().numpy()
    stdev = stdev_spread * (x_np.max() - x_np.min())
    total_grads = np.zeros_like(x_np)

    for _ in range(num_samples):
        noise = np.random.normal(0, stdev, x_np.shape).astype(np.float32)
        x_noisy = torch.from_numpy(x_np + noise).unsqueeze(0).to(device).requires_grad_(True)
        logits = model(x_noisy)
        target = logits[0, target_class_idx]
        model.zero_grad()
        target.backward()
        total_grads += np.abs(x_noisy.grad.detach().cpu().squeeze(0).numpy())

    return total_grads / num_samples
