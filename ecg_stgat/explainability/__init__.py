"""
ECG ST-GAT Explainability Package.
"""
from ecg_stgat.explainability.saliency import compute_gradient_saliency, compute_smoothgrad
from ecg_stgat.explainability.lead_importance import extract_lead_importance
from ecg_stgat.explainability.waveform_alignment import segment_ecg_intervals, compute_interval_alignment
from ecg_stgat.explainability.run_explainability import run_explainability_pipeline

__all__ = [
    "compute_gradient_saliency",
    "compute_smoothgrad",
    "extract_lead_importance",
    "segment_ecg_intervals",
    "compute_interval_alignment",
    "run_explainability_pipeline",
]
