"""
ECG Waveform Interval Segmentation & Saliency Alignment Metric (RQ3 / Table 5).
Measures saliency attribution mass concentrated inside QRS, ST-T, and P-wave intervals
using NeuroKit2 wave delineation, with explicit logging and fallback heuristics.
"""
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from scipy.signal import find_peaks
import neurokit2 as nk

logger = logging.getLogger(__name__)


def segment_ecg_intervals(lead_signal: np.ndarray, fs: float = 100.0) -> Dict[str, Any]:
    """
    Identifies sample-accurate boolean masks for P-wave, QRS complex, ST segment,
    and T-wave intervals using NeuroKit2 Discrete Wavelet Transform (DWT) delineation.
    If delineation cannot identify valid fiducial boundaries, falls back gracefully
    to peak-anchored heuristic windows and explicitly records the fallback status.

    Args:
        lead_signal: 1D array of shape (T,) representing an ECG lead waveform.
        fs: Sampling rate in Hz (default 100.0 Hz).

    Returns:
        Dict containing:
            'p_wave': np.ndarray (bool) of shape (T,)
            'qrs': np.ndarray (bool) of shape (T,)
            'st_segment': np.ndarray (bool) of shape (T,)
            't_wave': np.ndarray (bool) of shape (T,)
            'st_t': np.ndarray (bool) of shape (T,) [st_segment | t_wave]
            'baseline': np.ndarray (bool) of shape (T,) [isoelectric remainder]
            'peaks': np.ndarray of R-peak sample indices
            'delineation_success': bool
            'delineation_method': str ('neurokit2_dwt' or 'heuristic_fallback')
            'num_delineated_beats': int
            'failure_reason': Optional[str]
    """
    T = len(lead_signal)
    p_mask = np.zeros(T, dtype=bool)
    qrs_mask = np.zeros(T, dtype=bool)
    st_mask = np.zeros(T, dtype=bool)
    t_mask = np.zeros(T, dtype=bool)

    delineation_success = False
    method_used = "heuristic_fallback"
    valid_beats = 0
    failure_reason = None
    r_peaks_list = []

    # 1. Primary: Attempt NeuroKit2 DWT delineation
    try:
        _, rpeaks_dict = nk.ecg_peaks(lead_signal, sampling_rate=int(fs))
        r_peaks_list = rpeaks_dict.get("ECG_R_Peaks", [])

        if len(r_peaks_list) >= 1:
            _, waves = nk.ecg_delineate(lead_signal, rpeaks_dict, sampling_rate=int(fs), method="dwt")

            p_onsets = waves.get("ECG_P_Onsets", [])
            p_offsets = waves.get("ECG_P_Offsets", [])
            r_onsets = waves.get("ECG_R_Onsets", [])
            r_offsets = waves.get("ECG_R_Offsets", [])
            t_onsets = waves.get("ECG_T_Onsets", [])
            t_offsets = waves.get("ECG_T_Offsets", [])

            num_beats = len(r_peaks_list)
            for b in range(num_beats):
                beat_valid = False

                # QRS complex
                if (b < len(r_onsets) and b < len(r_offsets) and 
                        not np.isnan(r_onsets[b]) and not np.isnan(r_offsets[b])):
                    q_start = max(0, int(r_onsets[b]))
                    q_end = min(T, int(r_offsets[b]))
                    if q_start < q_end:
                        qrs_mask[q_start:q_end] = True
                        beat_valid = True

                # P-wave
                if (b < len(p_onsets) and b < len(p_offsets) and 
                        not np.isnan(p_onsets[b]) and not np.isnan(p_offsets[b])):
                    p_start = max(0, int(p_onsets[b]))
                    p_end = min(T, int(p_offsets[b]))
                    if p_start < p_end:
                        p_mask[p_start:p_end] = True
                        beat_valid = True

                # T-wave
                if (b < len(t_onsets) and b < len(t_offsets) and 
                        not np.isnan(t_onsets[b]) and not np.isnan(t_offsets[b])):
                    t_start = max(0, int(t_onsets[b]))
                    t_end = min(T, int(t_offsets[b]))
                    if t_start < t_end:
                        t_mask[t_start:t_end] = True
                        beat_valid = True

                # ST segment: interval between QRS offset (J-point) and T-wave onset
                if (b < len(r_offsets) and b < len(t_onsets) and 
                        not np.isnan(r_offsets[b]) and not np.isnan(t_onsets[b])):
                    st_start = int(r_offsets[b])
                    st_end = int(t_onsets[b])
                    if 0 <= st_start < st_end <= T:
                        st_mask[st_start:st_end] = True
                elif (b < len(r_offsets) and not np.isnan(r_offsets[b])):
                    # ST segment fallback if T onset missing: J-point to J + 100ms
                    st_start = max(0, int(r_offsets[b]))
                    st_end = min(T, st_start + int(0.10 * fs))
                    if st_start < st_end:
                        st_mask[st_start:st_end] = True

                if beat_valid:
                    valid_beats += 1

            if valid_beats >= 1:
                delineation_success = True
                method_used = "neurokit2_dwt"
        else:
            failure_reason = "No R-peaks detected by NeuroKit2"

    except Exception as exc:
        failure_reason = f"NeuroKit2 delineation exception: {str(exc)}"

    # 2. Fallback: Peak-anchored clinical window heuristics if delineation failed
    if not delineation_success:
        min_dist = int(0.4 * fs)
        peaks, _ = find_peaks(lead_signal, distance=min_dist, prominence=0.35 * np.std(lead_signal))
        if len(peaks) == 0:
            peaks, _ = find_peaks(lead_signal, distance=min_dist)
        r_peaks_list = list(peaks)

        for r in r_peaks_list:
            # QRS: ~ [-40ms, +50ms] -> [-4 samples, +5 samples at 100Hz]
            qrs_start = max(0, r - int(0.04 * fs))
            qrs_end = min(T, r + int(0.05 * fs))
            qrs_mask[qrs_start:qrs_end] = True

            # ST Segment: ~ [+50ms, +150ms]
            st_start = min(T, r + int(0.05 * fs))
            st_end = min(T, r + int(0.15 * fs))
            st_mask[st_start:st_end] = True

            # T-Wave: ~ [+150ms, +350ms]
            t_start = min(T, r + int(0.15 * fs))
            t_end = min(T, r + int(0.35 * fs))
            t_mask[t_start:t_end] = True

            # P-Wave: ~ [-200ms, -60ms]
            p_start = max(0, r - int(0.20 * fs))
            p_end = max(0, r - int(0.06 * fs))
            p_mask[p_start:p_end] = True

        valid_beats = len(r_peaks_list)

    st_t_mask = st_mask | t_mask
    baseline_mask = ~(p_mask | qrs_mask | st_mask | t_mask)

    return {
        "p_wave": p_mask,
        "qrs": qrs_mask,
        "st_segment": st_mask,
        "t_wave": t_mask,
        "st_t": st_t_mask,
        "baseline": baseline_mask,
        "peaks": np.array(r_peaks_list),
        "delineation_success": delineation_success,
        "delineation_method": method_used,
        "num_delineated_beats": valid_beats,
        "failure_reason": failure_reason,
    }


def compute_interval_alignment(
    saliency: np.ndarray,
    ecg_signal: np.ndarray,
    fs: float = 100.0,
) -> Dict[str, Any]:
    """
    Computes percentage of attribution mass falling in each clinical waveform interval.
    Args:
        saliency: (12, T) or (T,) non-negative gradient attribution scores
        ecg_signal: (12, T) or (T,) raw/preprocessed ECG waveform
        fs: Sampling rate in Hz (default 100 Hz)
    Returns:
        Dict with overlap percentages and delineation metadata.
    """
    if saliency.ndim == 2:
        sal = saliency.mean(axis=0)
        sig = ecg_signal[1] if ecg_signal.shape[0] > 1 else ecg_signal.mean(axis=0)
    else:
        sal = saliency
        sig = ecg_signal

    seg = segment_ecg_intervals(sig, fs=fs)
    total_mass = float(np.sum(sal)) + 1e-9

    qrs_mass = float(np.sum(sal[seg["qrs"]])) / total_mass * 100.0
    st_mass = float(np.sum(sal[seg["st_segment"]])) / total_mass * 100.0
    t_mass = float(np.sum(sal[seg["t_wave"]])) / total_mass * 100.0
    st_t_mass = float(np.sum(sal[seg["st_t"]])) / total_mass * 100.0
    p_mass = float(np.sum(sal[seg["p_wave"]])) / total_mass * 100.0
    base_mass = float(np.sum(sal[seg["baseline"]])) / total_mass * 100.0

    return {
        "qrs_overlap_pct": qrs_mass,
        "st_segment_overlap_pct": st_mass,
        "t_wave_overlap_pct": t_mass,
        "st_t_overlap_pct": st_t_mass,
        "p_wave_overlap_pct": p_mass,
        "baseline_overlap_pct": base_mass,
        "delineation_success": seg["delineation_success"],
        "delineation_method": seg["delineation_method"],
        "num_delineated_beats": seg["num_delineated_beats"],
        "failure_reason": seg["failure_reason"],
        "masks": seg,
    }


def compute_attribution_faithfulness(
    model: Any,
    ecg_tensor: Any,
    saliency: np.ndarray,
    target_class_idx: int,
    top_k_pct: float = 0.15,
) -> Dict[str, float]:
    """
    Quantifies attribution faithfulness via feature ablation.
    Masking the top-k% highest attribution samples should drop the target logit
    significantly more than masking random time samples.
    """
    import torch
    model.eval()
    if ecg_tensor.dim() == 2:
        ecg_tensor = ecg_tensor.unsqueeze(0)

    device = next(model.parameters()).device
    with torch.no_grad():
        orig_logit = float(model(ecg_tensor.to(device))[0, target_class_idx].cpu().item())

        # Salient masking
        T = ecg_tensor.shape[-1]
        k_samples = max(1, int(top_k_pct * T))
        sal_lead_mean = saliency.mean(axis=0) if saliency.ndim == 2 else saliency
        top_indices = np.argsort(sal_lead_mean)[-k_samples:]

        masked_sal = ecg_tensor.clone()
        masked_sal[:, :, top_indices] = 0.0
        sal_logit = float(model(masked_sal.to(device))[0, target_class_idx].cpu().item())

        # Random masking control
        rand_indices = np.random.choice(T, size=k_samples, replace=False)
        masked_rand = ecg_tensor.clone()
        masked_rand[:, :, rand_indices] = 0.0
        rand_logit = float(model(masked_rand.to(device))[0, target_class_idx].cpu().item())

    sal_drop = orig_logit - sal_logit
    rand_drop = orig_logit - rand_logit

    return {
        "original_logit": orig_logit,
        "salient_ablation_logit": sal_logit,
        "random_ablation_logit": rand_logit,
        "salient_drop": sal_drop,
        "random_drop": rand_drop,
        "faithfulness_ratio": float(sal_drop / max(rand_drop, 1e-4)),
    }
