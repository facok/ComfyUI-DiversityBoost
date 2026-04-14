"""DiversityBoost — Frequency-domain phase injection for composition diversity.

Restores composition diversity lost during distillation by injecting the initial
noise's low-frequency phase into the model's denoised prediction.  Operates at
step 0 only via a post-CFG hook — completely external, zero risk to model internals.

Core idea: in frequency domain, amplitude encodes energy distribution ("naturalness")
while phase encodes spatial arrangement ("composition").  By rotating the prediction's
low-frequency phase toward the noise's phase — while attenuating high-frequency
amplitude to prevent "frequency shearing" — we restore per-seed composition diversity
that distillation collapsed.

Key design decisions:
  - Slerp (spherical interpolation): rotate pred's phase toward noise's phase by
    γ·θ on the unit circle.  Unlike lerp+renorm, Δφ is strictly linear in strength
    — no topological singularity when opposing phasors cancel at γ≈0.5.
  - Shared rotation field: compute a single θ per spatial-freq bin from channel-mean
    phasors of pred and noise, then broadcast to all channels.  Inter-channel
    relative phase is exactly preserved → zero chromatic fringing.
  - Per-bin rotation budget (φ_max): θ ~ Uniform(-π,π) has CV=1/√3 ≈ 58%, so
    the tail (max rotation) is always 2× the mean.  Without a cap, strength
    simultaneously controls diversity and tail risk at a fixed ratio — making
    the usable range extremely narrow ("weak → sudden collapse").  The tanh
    soft-cap φ_eff = φ_max·tanh(φ_raw/φ_max) decouples mean diversity from
    worst-case rotation, widening the usable strength range to [0, 1].
  - 6th-order Butterworth LPF (not Gaussian): steep transition band cleanly
    separates composition-scale frequencies (≤4 spatial periods across the frame)
    from object-scale frequencies (≥5 periods).  Uses elliptical normalization
    so n_periods applies independently per axis — resolution and aspect-ratio
    independent.  A Gaussian with σ=0.05 leaks 0.73 at r=0.04, causing
    probabilistic distortion of object structure.
  - High-frequency attenuation (hf_preserve): distilled models produce a
    fully-formed x0_hat at step 0 where HF details are spatially bound to LF
    structure.  Rotating only LF phase while preserving HF amplitude causes
    "frequency shearing" — the body moves but fingers/edges stay pinned at
    the original position, forcing the model to generate deformities.
    Attenuating HF amplitude makes x0_hat resemble a teacher's blurry step-0
    prediction, letting subsequent steps freely reconstruct coherent details.
"""

import logging
import math

import torch

from .sampling import (
    denoised_to_raw,
    raw_to_denoised,
    find_step_index,
    unpack_video_if_needed,
    repack_video_if_needed,
)

log = logging.getLogger("ComfyUI-DiversityBoost")


def _build_freq_mask(H, W, n_periods, device):
    """Build an elliptical Butterworth low-pass frequency mask for rfft2 output.

    Returns a real-valued mask [1, 1, H, W//2+1] with:
    - 6th-order Butterworth rolloff per-axis normalized
    - DC component zeroed (preserves image mean / energy conservation)

    Uses elliptical normalization: each axis is normalized by its own cutoff
    (n_periods/H for vertical, n_periods/W for horizontal).  This ensures
    "n_periods=2" means ≤2 full cycles in EACH direction, regardless of
    aspect ratio.  A circular mask with cutoff = n/max(H,W) would under-count
    bins along the shorter axis.
    """
    freq_y = torch.fft.fftfreq(H, device=device).unsqueeze(1)     # [H, 1]
    freq_x = torch.fft.rfftfreq(W, device=device).unsqueeze(0)    # [1, W//2+1]

    # Elliptical normalized radius: each axis scaled by its own dimension.
    # r_norm = sqrt((fy * H / n)^2 + (fx * W / n)^2)
    # r_norm = 1.0 at the cutoff boundary in each axis direction.
    r_norm = torch.sqrt((freq_y * H / n_periods) ** 2 +
                        (freq_x * W / n_periods) ** 2)

    # 6th-order Butterworth: 1 / sqrt(1 + r_norm^12).
    # r_norm = 1 → mask = 0.707 (cutoff), r_norm = 2 → mask ≈ 0.016.
    mask = 1.0 / torch.sqrt(1.0 + r_norm.pow(12))

    # Zero DC: preserve mean intensity
    mask[0, 0] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W//2+1]


def _phase_inject(raw_pred, raw_noise, strength, freq_mask, max_rotation,
                   hf_preserve, energy_compensate, dc_preserve):
    """Low-frequency phase injection with optional high-frequency attenuation.

    Given:
        raw_pred      — model's denoised prediction x0_hat  [B, C, H, W]
        raw_noise     — initial noise z_T (≈ args["input"] at step 0)  [B, C, H, W]
        strength      — blend factor gamma in [0, 1]
        freq_mask     — Butterworth LPF mask from _build_freq_mask  [1, 1, H, W//2+1]
        max_rotation  — per-bin rotation budget φ_max in radians (>0), or 0 to disable
        hf_preserve   — high-frequency amplitude preservation factor D in [0, 1].
                         1.0 = preserve all HF amplitude (original behavior).
                         0.0 = attenuate HF amplitude to zero (maximum blur).
        energy_compensate — if True, rescale output RMS to match original prediction.
        dc_preserve   — DC amplitude preservation factor in [0, 1].
                         0.0 = DC attenuated like HF (model rebuilds brightness).
                         1.0 = DC fully preserved (original brightness).

    Returns:
        raw_new — modified prediction with injected phase  [B, C, H, W]
    """
    H, W = raw_pred.shape[-2:]

    # Step 1: Forward FFT
    F_pred = torch.fft.rfft2(raw_pred.float())    # [B, C, H, W//2+1] complex
    F_noise = torch.fft.rfft2(raw_noise.float())   # [B, C, H, W//2+1] complex

    # Step 2: Shared rotation field — compute a single θ per spatial-freq bin,
    # then broadcast to all channels.  This guarantees inter-channel relative
    # phase is EXACTLY preserved (identical rotation for all C channels at each bin).
    unit_pred_per_ch = F_pred / F_pred.abs().clamp(min=1e-8)             # [B, C, H, W//2+1]
    unit_pred_shared = unit_pred_per_ch.mean(dim=1, keepdim=True)        # [B, 1, H, W//2+1]
    unit_pred_shared = unit_pred_shared / unit_pred_shared.abs().clamp(min=1e-8)

    unit_noise_per_ch = F_noise / F_noise.abs().clamp(min=1e-8)
    unit_noise_shared = unit_noise_per_ch.mean(dim=1, keepdim=True)      # [B, 1, H, W//2+1]
    unit_noise_shared = unit_noise_shared / unit_noise_shared.abs().clamp(min=1e-8)

    # θ_shared: shortest arc from shared-pred to shared-noise phase.
    theta = (unit_pred_shared.conj() * unit_noise_shared).angle()        # [B, 1, H, W//2+1]

    # Step 3: Per-bin rotation with tanh soft-cap (rotation budget).
    gamma_mask = strength * freq_mask                                   # [1, 1, H, W//2+1]
    phi_raw = gamma_mask * theta                                        # [B, 1, H, W//2+1]

    if max_rotation > 0:
        phi_eff = max_rotation * torch.tanh(phi_raw / max_rotation)     # [B, 1, H, W//2+1]
    else:
        phi_eff = phi_raw  # no cap — original behavior

    rotation = torch.exp(1j * phi_eff)                                  # [B, 1, H, W//2+1]

    # Step 4: Apply rotation + optional high-frequency amplitude attenuation.
    # S(r) = M(r) + D·(1 - M(r)):  low-freq bins (M≈1) get S≈1 (preserved),
    # high-freq bins (M≈0) get S=D (attenuated).
    if hf_preserve < 1.0 - 1e-6:
        amp_scale = freq_mask + hf_preserve * (1.0 - freq_mask)         # [1, 1, H, W//2+1]
        # DC amplitude: dc_preserve controls how much DC is kept.
        # freq_mask[0,0]=0 → amp_scale[0,0]=hf_preserve without this line.
        # dc_preserve=1 → fully preserve, dc_preserve=0 → same as hf_preserve.
        amp_scale[:, :, 0, 0] = dc_preserve
        F_new = F_pred * (rotation * amp_scale)
    else:
        F_new = F_pred * rotation  # amplitude exactly preserved

    # Step 5: Inverse FFT back to spatial domain
    raw_new = torch.fft.irfft2(F_new, s=(H, W))

    # Optional energy compensation: when hf_preserve < 1, amp_scale reduces
    # total energy.  This rescales per-sample RMS back to the original level.
    if energy_compensate and hf_preserve < 1.0 - 1e-6:
        pred_rms = raw_pred.float().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
        new_rms = raw_new.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
        raw_new = raw_new * (pred_rms / new_rms)

    return raw_new.to(dtype=raw_pred.dtype)


def build_phase_injection_fn(strength=1.0, n_periods=2, max_rotation=1.5708,
                             hf_preserve=0.0, energy_compensate=False,
                             dc_preserve=0.0):
    """Build a post_cfg_function that injects noise phase into the denoised prediction.

    Parameters:
        strength: blend factor gamma (0 = no effect, 1 = full injection).
        n_periods: max spatial periods to affect (integer).
                   The Butterworth cutoff uses elliptical normalization:
                   r_norm = sqrt((ky/n)^2 + (kx/n)^2), where ky,kx are
                   integer cycle counts per axis.  Resolution and aspect-ratio
                   independent — n_periods=2 always covers the same bins.
                   - 1: ultra-conservative (global balance only)
                   - 2: conservative (default, recommended)
                   - 3: balanced (moderate diversity)
                   - 4: aggressive (more diversity, higher risk)
        max_rotation: per-bin rotation budget φ_max in radians (0 = no cap).
                      Default π/2 ≈ 1.5708.
        hf_preserve: high-frequency amplitude preservation factor D in [0, 1].
                     D=0 produces a blurry "composition sketch", D=1 preserves all HF.
        energy_compensate: if True, rescale output RMS to match original prediction.
        dc_preserve: DC amplitude preservation factor in [0, 1].
                     0 = DC attenuated (model rebuilds brightness freely).
                     1 = DC fully preserved (original brightness kept).

    Returns:
        A closure suitable for model.set_model_sampler_post_cfg_function().
    """

    state = {
        "freq_mask": None,
        "cached_hw": None,
    }

    def phase_inject_hook(args):
        denoised = args["denoised"]
        sigma = args["sigma"]
        model = args["model"]
        model_options = args["model_options"]
        x_t = args["input"]

        # --- Step 0 only ---
        sample_sigmas = model_options.get("transformer_options", {}).get("sample_sigmas")
        if sample_sigmas is None:
            return denoised

        step_index = find_step_index(sigma, sample_sigmas)
        if step_index != 0:
            return denoised

        # --- Handle LTXAV packed format ---
        working, pack_info = unpack_video_if_needed(denoised, args)
        x_t_working, _ = unpack_video_if_needed(x_t, args)

        # --- Convert to raw space (undo process_in) ---
        raw_pred = denoised_to_raw(working, model)
        raw_noise = denoised_to_raw(x_t_working, model)

        # --- Build or reuse frequency mask ---
        H, W = raw_pred.shape[-2:]
        if state["cached_hw"] != (H, W):
            state["freq_mask"] = _build_freq_mask(H, W, n_periods, raw_pred.device)
            state["cached_hw"] = (H, W)
        freq_mask = state["freq_mask"].to(device=raw_pred.device)

        # --- Core: phase injection ---
        raw_new = _phase_inject(raw_pred, raw_noise, strength, freq_mask,
                                max_rotation, hf_preserve, energy_compensate,
                                dc_preserve)

        # --- Logging ---
        if log.isEnabledFor(logging.INFO):
            with torch.no_grad():
                delta = (raw_new - raw_pred).float()
                B = raw_pred.shape[0]

                # Per-batch-item diagnostics
                per_item = []
                for b in range(B):
                    d_rms = delta[b].pow(2).mean().sqrt().item()
                    p_rms = raw_pred[b].float().pow(2).mean().sqrt().item()
                    per_item.append((d_rms, p_rms))

                delta_rms = delta.pow(2).mean().sqrt().item()
                pred_rms = raw_pred.float().pow(2).mean().sqrt().item()

                if B > 1:
                    per_item_str = "  ".join(
                        f"b{b}={d:.4f}/{p:.4f}" for b, (d, p) in enumerate(per_item)
                    )
                    log.info(
                        "[DiversityBoost] step=0  strength=%.3f  n_periods=%d  "
                        "max_rotation=%.3f  hf_preserve=%.3f  "
                        "shape=%s  delta_rms=%.4f  pred_rms=%.4f  ratio=%.4f  "
                        "per_batch(delta/pred): %s",
                        strength, n_periods,
                        max_rotation, hf_preserve,
                        list(raw_pred.shape),
                        delta_rms, pred_rms,
                        delta_rms / max(pred_rms, 1e-8),
                        per_item_str,
                    )
                else:
                    log.info(
                        "[DiversityBoost] step=0  strength=%.3f  n_periods=%d  "
                        "max_rotation=%.3f  hf_preserve=%.3f  "
                        "shape=%s  delta_rms=%.4f  pred_rms=%.4f  ratio=%.4f",
                        strength, n_periods,
                        max_rotation, hf_preserve,
                        list(raw_pred.shape),
                        delta_rms, pred_rms,
                        delta_rms / max(pred_rms, 1e-8),
                    )

        # --- Convert back to process_in space ---
        modified = raw_to_denoised(raw_new, model).to(dtype=denoised.dtype)

        return repack_video_if_needed(modified, pack_info)

    return phase_inject_hook
