"""DiversityBoost core — HF attenuation + DCT composition push.

Restores composition diversity lost during distillation via two mechanisms
applied in a single post-cfg hook at step 0:
1. HF attenuation (Butterworth LPF) — blurry "composition sketch"
2. DCT composition push — multiplicative low-freq spatial field
"""

import logging
import math
from functools import lru_cache

import torch

from .sampling import (
    denoised_to_raw,
    raw_to_denoised,
    find_step_index,
    unpack_video_if_needed,
    repack_video_if_needed,
)

log = logging.getLogger("ComfyUI-DiversityBoost")


# ---------------------------------------------------------------------------
# 2D DCT basis (orthonormal, cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _build_dct_basis_2d(H, W, n_modes_h=4, n_modes_w=4):
    """Build orthonormal 2D DCT-II basis matrix [H*W, n_modes_h*n_modes_w].

    Same matrix for analysis (projection) and synthesis (reconstruction):
      synthesis: field = basis @ coeffs
    """
    def _dct1d(N, n_modes):
        n = torch.arange(N, dtype=torch.float64)
        k = torch.arange(n_modes, dtype=torch.float64)
        phi = torch.cos(math.pi * k[None, :] * (n[:, None] + 0.5) / N)
        norm = torch.full((n_modes,), math.sqrt(2.0 / N), dtype=torch.float64)
        norm[0] = 1.0 / math.sqrt(N)
        return phi * norm[None, :]

    phi_h = _dct1d(H, n_modes_h)
    phi_w = _dct1d(W, n_modes_w)
    basis_2d = torch.einsum('hu,wv->hwuv', phi_h, phi_w)
    basis_2d = basis_2d.reshape(H * W, n_modes_h * n_modes_w)
    return basis_2d.float()


# ---------------------------------------------------------------------------
# Noise frequency weights
# ---------------------------------------------------------------------------

def _build_pink_weights(n_h, n_w):
    """1/f amplitude weights: lower frequencies dominate."""
    weights = []
    for u in range(n_h):
        for v in range(n_w):
            freq_sq = u * u + v * v
            weights.append(0.0 if freq_sq == 0 else 1.0 / (freq_sq ** 0.25))
    return torch.tensor(weights, dtype=torch.float32)


def _build_blue_weights(n_h, n_w):
    """f-proportional weights: higher frequencies dominate."""
    weights = []
    for u in range(n_h):
        for v in range(n_w):
            freq_sq = u * u + v * v
            weights.append(0.0 if freq_sq == 0 else (freq_sq ** 0.25))
    return torch.tensor(weights, dtype=torch.float32)


def _build_noise_weights(noise_type, n_h, n_w):
    """Build frequency weights for given noise type, or None for white."""
    if noise_type == "pink":
        return _build_pink_weights(n_h, n_w)
    elif noise_type == "blue":
        return _build_blue_weights(n_h, n_w)
    return None


# ---------------------------------------------------------------------------
# Butterworth LPF
# ---------------------------------------------------------------------------

def _build_freq_mask(H, W, n_periods, device):
    """Elliptical Butterworth LPF mask for rfft2 output [1, 1, H, W//2+1]."""
    freq_y = torch.fft.fftfreq(H, device=device).unsqueeze(1)
    freq_x = torch.fft.rfftfreq(W, device=device).unsqueeze(0)
    r_norm = torch.sqrt((freq_y * H / n_periods) ** 2 +
                        (freq_x * W / n_periods) ** 2)
    mask = 1.0 / torch.sqrt(1.0 + r_norm.pow(12))
    mask[0, 0] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Combined hook: HF attenuation → DCT composition push
# ---------------------------------------------------------------------------

def build_diversity_fn(strength=0.5, clamp_val=1.0, noise_type="pink",
                       n_periods=2, dc_preserve=0.0,
                       energy_compensate=False):
    """Build a post_cfg_function that attenuates HF then applies DCT push.

    Execution order within step 0:
      1. Convert to raw latent space
      2. HF attenuation (Butterworth LPF)
      3. DCT composition push (4×4 random spatial field, multiplicative)
      4. Convert back

    The push operates on the blurred x0_hat in latent pixel space [B,C,H,W],
    not token space.  This ensures nothing downstream can erase the signal.

    Parameters:
        strength:          push amplitude (0-2). 0 = no push (cleanup only).
        clamp_val:         safety clamp for field values.
        noise_type:        "pink", "white", or "blue" frequency weighting.
        n_periods:         Butterworth cutoff (spatial periods to preserve).
        dc_preserve:       DC amplitude preservation [0, 1].
        energy_compensate: rescale output RMS to match original prediction.
    """
    n_modes_h, n_modes_w = 4, 4
    n_modes = n_modes_h * n_modes_w
    freq_weights = _build_noise_weights(noise_type, n_modes_h, n_modes_w)

    state = {
        "amp_scale": None,
        "basis_2d": None,
        "freq_weights": None,
        "cached_hw": None,
    }

    def diversity_hook(args):
        denoised = args["denoised"]
        sigma = args["sigma"]
        model = args["model"]
        model_options = args["model_options"]

        # --- Step 0 only ---
        sample_sigmas = model_options.get("transformer_options", {}).get("sample_sigmas")
        if sample_sigmas is None:
            return denoised
        step_index = find_step_index(sigma, sample_sigmas)
        if step_index != 0:
            return denoised

        # --- Unpack video if needed ---
        working, pack_info = unpack_video_if_needed(denoised, args)

        # --- Convert to raw space ---
        raw_pred = denoised_to_raw(working, model)
        B, C, H, W = raw_pred.shape
        device = raw_pred.device
        orig_dtype = raw_pred.dtype

        # --- Build or reuse cached tensors ---
        if state["cached_hw"] != (H, W):
            freq_mask = _build_freq_mask(H, W, n_periods, device)
            amp = freq_mask.clone()
            amp[:, :, 0, 0] = dc_preserve
            state["amp_scale"] = amp
            if strength > 1e-6:
                basis_cpu = _build_dct_basis_2d(H, W, n_modes_h, n_modes_w)
                state["basis_2d"] = basis_cpu.to(device=device)
                if freq_weights is not None:
                    state["freq_weights"] = freq_weights.to(device=device)
            state["cached_hw"] = (H, W)
        amp_scale = state["amp_scale"].to(device=device)

        # --- Step 1: HF attenuation ---
        F_pred = torch.fft.rfft2(raw_pred.float())
        F_blurred = F_pred * amp_scale
        raw_blurred = torch.fft.irfft2(F_blurred, s=(H, W))

        # --- Step 2: DCT composition push on blurred result ---
        if strength > 1e-6:
            coeffs = torch.randn(B, n_modes, device=device, dtype=torch.float32)
            coeffs[:, 0] = 0.0

            if state["freq_weights"] is not None:
                coeffs = coeffs * state["freq_weights"]

            field = torch.einsum('nk,bk->bn', state["basis_2d"], coeffs)
            field = field.reshape(B, H, W)

            field_std = field.reshape(B, -1).std(dim=1).clamp(min=1e-8)
            field = field / field_std[:, None, None]
            field = field * strength

            scale = (1.0 + field).clamp(min=0.10, max=1.0 + clamp_val).unsqueeze(1)
            raw_new = raw_blurred * scale
        else:
            raw_new = raw_blurred
            scale = None

        # --- Energy compensation ---
        if energy_compensate:
            pred_rms = raw_pred.float().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
            new_rms = raw_new.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
            raw_new = raw_new * (pred_rms / new_rms)

        # --- Logging ---
        if log.isEnabledFor(logging.INFO):
            with torch.no_grad():
                delta = (raw_new - raw_pred.float())
                delta_rms = delta.pow(2).mean().sqrt().item()
                pred_rms_val = raw_pred.float().pow(2).mean().sqrt().item()
                push_info = ""
                if scale is not None:
                    s_flat = scale.squeeze(1)
                    push_info = (
                        f"  push=[{s_flat.min().item():.4f}, {s_flat.max().item():.4f}]"
                        f"  strength={strength:.3f}  clamp={clamp_val:.2f}  noise={noise_type}"
                    )
                log.info(
                    "[DiversityBoost] step=0  n_periods=%d  dc=%.2f"
                    "%s  shape=%s  delta_rms=%.4f  pred_rms=%.4f  ratio=%.4f",
                    n_periods, dc_preserve,
                    push_info, list(raw_pred.shape),
                    delta_rms, pred_rms_val,
                    delta_rms / max(pred_rms_val, 1e-8),
                )

        # --- Convert back ---
        modified = raw_to_denoised(raw_new, model).to(dtype=orig_dtype)
        return repack_video_if_needed(modified, pack_info)

    return diversity_hook
