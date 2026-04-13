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
    from object-scale frequencies (≥5 periods).  At r_cut=0.02 (default):
    bins with r ≤ 0.01 get mask ≈ 1.0 (full composition diversity), bins at
    r = 0.04 get mask ≈ 0.02 (object internals fully protected).  A Gaussian
    with σ=0.05 leaks 0.73 at r=0.04, causing probabilistic distortion of
    object structure.
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


def _build_freq_mask(H, W, freq_cutoff, device):
    """Build a Butterworth low-pass frequency mask for rfft2 output.

    Returns a real-valued mask [1, 1, H, W//2+1] with:
    - 6th-order Butterworth rolloff: 1 / sqrt(1 + (r/r_cut)^12)
    - DC component zeroed (preserves image mean / energy conservation)

    6th-order Butterworth is chosen for its steep transition band that cleanly
    separates composition-scale frequencies from object-scale frequencies.
    At the default r_cut=0.02, the profile is:
        r ≤ 0.01 (≤2 spatial periods): mask ≈ 1.0  (full composition diversity)
        r = 0.02 (4 periods, cutoff):   mask = 0.71 (transition band)
        r ≥ 0.04 (≥8 periods):          mask ≈ 0.02 (object internals protected)

    A Gaussian cannot achieve this: at σ=0.05, r=0.04 gets mask=0.73 (causes
    probabilistic object distortion).  Lower-order Butterworth (n=4) either
    affects too few bins (steep cutoff) or leaks into object scale.
    """
    freq_y = torch.fft.fftfreq(H, device=device).unsqueeze(1)     # [H, 1]
    freq_x = torch.fft.rfftfreq(W, device=device).unsqueeze(0)    # [1, W//2+1]

    freq_r = torch.sqrt(freq_y ** 2 + freq_x ** 2)

    # 6th-order Butterworth: 1 / sqrt(1 + (r/r_c)^(2n)), n=6 → exponent=12.
    # Maximally flat passband, steep rolloff, no ringing.
    # At r = 2*r_cut: mask = 1/sqrt(1 + 4096) ≈ 0.016.
    mask = 1.0 / torch.sqrt(1.0 + (freq_r / freq_cutoff).pow(12))

    # Zero DC: preserve mean intensity
    mask[0, 0] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W//2+1]


def _phase_inject(raw_pred, raw_noise, strength, freq_mask, max_rotation,
                   hf_preserve, energy_compensate):
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
        # DC bin must be preserved: freq_mask zeroes DC (for phase rotation),
        # but amp_scale must NOT zero DC amplitude (that's the image mean).
        amp_scale[:, :, 0, 0] = 1.0
        F_new = F_pred * (rotation * amp_scale)
    else:
        F_new = F_pred * rotation  # amplitude exactly preserved

    # Step 5: Inverse FFT back to spatial domain
    raw_new = torch.fft.irfft2(F_new, s=(H, W))

    # Optional energy compensation: when hf_preserve < 1, amp_scale reduces
    # total energy.  This rescales per-sample RMS back to the original level.
    if energy_compensate and hf_preserve < 1.0 - 1e-6:
        pred_rms = raw_pred.float().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
        new_rms = raw_new.float().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
        raw_new = raw_new * (pred_rms / new_rms)

    return raw_new.to(dtype=raw_pred.dtype)


def build_phase_injection_fn(strength=1.0, freq_cutoff=0.02, max_rotation=1.5708,
                             hf_preserve=0.0, energy_compensate=False):
    """Build a post_cfg_function that injects noise phase into the denoised prediction.

    Parameters:
        strength: blend factor gamma (0 = no effect, 1 = full injection).
        freq_cutoff: Butterworth cutoff in normalized frequency units.
                     Controls how many spatial periods are affected:
                     - 0.010: ≤2 periods (global balance only, ultra-conservative)
                     - 0.015: ≤3 periods (conservative, minimal object influence)
                     - 0.020: ≤4 periods (balanced, recommended)
                     - 0.025: ≤5 periods (aggressive, more diversity, higher risk)
        max_rotation: per-bin rotation budget φ_max in radians (0 = no cap).
                      Default π/2 ≈ 1.5708.
        hf_preserve: high-frequency amplitude preservation factor D in [0, 1].
                     D=0 produces a blurry "composition sketch", D=1 preserves all HF.
        energy_compensate: if True, rescale output RMS to match original prediction.

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
            state["freq_mask"] = _build_freq_mask(H, W, freq_cutoff, raw_pred.device)
            state["cached_hw"] = (H, W)
        freq_mask = state["freq_mask"].to(device=raw_pred.device)

        # --- Core: phase injection ---
        raw_new = _phase_inject(raw_pred, raw_noise, strength, freq_mask,
                                max_rotation, hf_preserve, energy_compensate)

        # --- Logging ---
        with torch.no_grad():
            delta = (raw_new - raw_pred).float()
            delta_rms = delta.pow(2).mean().sqrt().item()
            pred_rms = raw_pred.float().pow(2).mean().sqrt().item()

            if log.isEnabledFor(logging.DEBUG):
                F_pred_log = torch.fft.rfft2(raw_pred.float())
                F_new_log = torch.fft.rfft2(raw_new.float())
                phase_diff = (F_new_log * F_pred_log.conj()).angle().abs()
                mask_thresh = freq_mask > 0.1
                if mask_thresh.any():
                    mean_phase_shift = phase_diff[mask_thresh.expand_as(phase_diff)].mean().item()
                else:
                    mean_phase_shift = 0.0

                log.debug(
                    "[DiversityBoost] step=0  strength=%.3f  freq_cutoff=%.3f  "
                    "max_rotation=%.3f  hf_preserve=%.3f  shape=%s  "
                    "delta_rms=%.4f  pred_rms=%.4f  ratio=%.4f  "
                    "mean_phase_shift=%.3f rad (%.1f deg)",
                    strength, freq_cutoff, max_rotation, hf_preserve,
                    list(raw_pred.shape),
                    delta_rms, pred_rms,
                    delta_rms / max(pred_rms, 1e-8),
                    mean_phase_shift,
                    math.degrees(mean_phase_shift),
                )
            else:
                log.info(
                    "[DiversityBoost] step=0  strength=%.3f  freq_cutoff=%.3f  "
                    "max_rotation=%.3f  hf_preserve=%.3f  shape=%s  "
                    "delta_rms=%.4f  pred_rms=%.4f  ratio=%.4f",
                    strength, freq_cutoff, max_rotation, hf_preserve,
                    list(raw_pred.shape),
                    delta_rms, pred_rms,
                    delta_rms / max(pred_rms, 1e-8),
                )

        # --- Convert back to process_in space ---
        modified = raw_to_denoised(raw_new, model).to(dtype=denoised.dtype)

        return repack_video_if_needed(modified, pack_info)

    return phase_inject_hook
