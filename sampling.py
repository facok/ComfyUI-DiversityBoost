"""Shared sampling utilities for DiversityBoost hooks."""

import math

import comfy.utils
import torch


def find_step_index(sigma, sigmas):
    """Find the step index for a given sigma value in the sigma schedule.

    Uses torch.isclose for robust matching across dtype differences (e.g. bfloat16
    sigma vs float32 sample_sigmas), with argmin fallback for edge cases.
    """
    sigma_val = sigma.flatten()[0].float()
    sigmas_f = sigmas.float()
    matched = torch.isclose(sigmas_f, sigma_val, rtol=1e-3, atol=1e-5).nonzero()
    if len(matched) > 0:
        return matched[0].item()
    return (sigmas_f - sigma_val).abs().argmin().item()


def denoised_to_raw(denoised, model):
    """Convert denoised tensor from process_in space to raw VAE space."""
    return model.latent_format.process_out(denoised)


def raw_to_denoised(raw, model):
    """Convert raw VAE space tensor back to process_in space."""
    return model.latent_format.process_in(raw)


def unpack_video_if_needed(denoised, args):
    """Unpack LTXAV-style packed latents if detected.

    Returns (tensor_to_process, pack_info) where pack_info is None for
    non-packed formats or a dict for repacking.
    """
    if denoised.ndim == 3 and denoised.shape[1] == 1:
        cond = args.get("cond")
        latent_shapes = _extract_latent_shapes(cond)
        if latent_shapes is not None and len(latent_shapes) > 1:
            tensors = comfy.utils.unpack_latents(denoised, latent_shapes)
            return tensors[0], {"other_tensors": tensors[1:]}
    return denoised, None


def repack_video_if_needed(modified, pack_info):
    """Repack video tensor back into LTXAV packed format if it was unpacked."""
    if pack_info is None:
        return modified
    all_tensors = [modified] + pack_info["other_tensors"]
    packed, _ = comfy.utils.pack_latents(all_tensors)
    return packed


def _extract_latent_shapes(cond):
    """Try to extract latent_shapes from conditioning data."""
    if cond is None:
        return None
    for c in cond:
        if isinstance(c, dict):
            model_conds = c.get('model_conds', {})
            if 'latent_shapes' in model_conds:
                ls = model_conds['latent_shapes']
                if hasattr(ls, 'cond'):
                    return ls.cond
                return ls
    return None


# ---------------------------------------------------------------------------
# Shared saturation helper
# ---------------------------------------------------------------------------

def apply_saturation(field, mode, max_amp, hard_clamp_val, min_scale=0.10):
    """Map normalized field to multiplicative scale.

    mode="tanh": smooth symmetric saturation, scale ∈ [1-max_amp, 1+max_amp]
    mode="hard" : clamp(1+field, min_scale, 1+hard_clamp_val)
    """
    if mode == "tanh":
        return 1.0 + torch.tanh(field) * max_amp
    return (1.0 + field).clamp(min=min_scale, max=1.0 + hard_clamp_val)


# ---------------------------------------------------------------------------
# Pure PyTorch Haar DWT/IDWT
# ---------------------------------------------------------------------------

def dwt2_haar(x):
    """2D Haar DWT.  x: [B, C, H, W] where H, W are even.
    Returns ll, lh, hl, hh each [B, C, H//2, W//2].
    """
    x_l = (x[:, :, :, 0::2] + x[:, :, :, 1::2]) / math.sqrt(2.0)
    x_h = (x[:, :, :, 0::2] - x[:, :, :, 1::2]) / math.sqrt(2.0)
    ll = (x_l[:, :, 0::2, :] + x_l[:, :, 1::2, :]) / math.sqrt(2.0)
    lh = (x_l[:, :, 0::2, :] - x_l[:, :, 1::2, :]) / math.sqrt(2.0)
    hl = (x_h[:, :, 0::2, :] + x_h[:, :, 1::2, :]) / math.sqrt(2.0)
    hh = (x_h[:, :, 0::2, :] - x_h[:, :, 1::2, :]) / math.sqrt(2.0)
    return ll, lh, hl, hh


def idwt2_haar(ll, lh, hl, hh):
    """2D Haar IDWT.  Inverse of dwt2_haar."""
    B, C, H2, W2 = ll.shape
    # Vertical inverse: interleave rows
    x_l = torch.zeros(B, C, H2 * 2, W2, device=ll.device, dtype=ll.dtype)
    x_h = torch.zeros(B, C, H2 * 2, W2, device=hl.device, dtype=hl.dtype)
    x_l[:, :, 0::2, :] = (ll + lh) / math.sqrt(2.0)
    x_l[:, :, 1::2, :] = (ll - lh) / math.sqrt(2.0)
    x_h[:, :, 0::2, :] = (hl + hh) / math.sqrt(2.0)
    x_h[:, :, 1::2, :] = (hl - hh) / math.sqrt(2.0)
    # Horizontal inverse: interleave cols
    x = torch.zeros(B, C, H2 * 2, W2 * 2, device=x_l.device, dtype=x_l.dtype)
    x[:, :, :, 0::2] = (x_l + x_h) / math.sqrt(2.0)
    x[:, :, :, 1::2] = (x_l - x_h) / math.sqrt(2.0)
    return x


# ---------------------------------------------------------------------------
# Gaussian blur (separable, no Gibbs ringing)
# ---------------------------------------------------------------------------

def gaussian_blur_2d(x, sigma):
    """Separable Gaussian blur over last two spatial dims of [B, C, H, W]."""
    if sigma <= 0.0:
        return x
    radius = max(1, int(math.ceil(sigma * 3)))
    coords = torch.arange(2 * radius + 1, dtype=torch.float32) - radius
    g = torch.exp(-coords.pow(2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    g = g.to(device=x.device, dtype=x.dtype)
    C = x.shape[1]
    kx = g.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = g.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = torch.nn.functional.conv2d(x, kx, padding=(0, radius), groups=C)
    x = torch.nn.functional.conv2d(x, ky, padding=(radius, 0), groups=C)
    return x
