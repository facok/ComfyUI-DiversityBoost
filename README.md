# ComfyUI-DiversityBoost

Restore composition diversity for distilled diffusion models. Training-free, single-step frequency-domain phase injection.

> [中文版 README](README_zh.md)

## The Problem

Step-distilled models (FLUX2.[Klein], z-image-turbo, etc.) generate high-quality images in few steps, but suffer from **composition collapse**: different seeds produce nearly identical layouts. A portrait prompt always puts the subject dead-center; a landscape prompt always uses the same horizon line.

Root cause: distillation freezes the spatial distribution of token norms across seeds, locking the model into a single "average" composition regardless of the initial noise.

## The Fix

DiversityBoost injects seed-dependent low-frequency phase from the initial noise into the model's denoised prediction at **step 0 only**, via a post-CFG hook.

In the frequency domain, **amplitude** encodes energy distribution (naturalness), while **phase** encodes spatial arrangement (composition). By rotating only low-frequency phase — while attenuating high-frequency amplitude to prevent deformities — different seeds produce genuinely different compositions again.

Zero model modification. Zero training. One node.

## How It Works

1. **FFT** the model's step-0 prediction and the initial noise
2. Compute a **shared rotation field** (channel-mean phasors) — preserves inter-channel phase exactly (no color fringing)
3. Apply a **6th-order Butterworth low-pass filter** — only composition-scale frequencies are touched; object-scale details are protected
4. **Tanh soft-cap** per-bin rotation — decouples diversity strength from worst-case rotation risk
5. **Attenuate high-frequency amplitude** — prevents "frequency shearing" (body moves but fingers stay pinned)
6. **IFFT** back to spatial domain

Step 0 only. The model's own attractor handles the rest.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/facok/ComfyUI-DiversityBoost.git
```

No extra dependencies — only requires PyTorch (ships with ComfyUI).

## Usage

```
MODEL → [Diversity Boost] → MODEL → KSampler
```

1. Add the **Diversity Boost** node (category: `sampling`)
2. Connect your model to the input, output to KSampler
3. Generate with different seeds — compositions will vary

Default settings work well. No tuning needed for most use cases.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **strength** | 1.0 | 0.0 – 1.0 | Phase rotation strength. 1.0 is safe with the default max_rotation cap. |
| **n_periods** | 2 | 1 – 10 | Max spatial periods to affect. Resolution-independent. |
| **max_rotation** | 1.5708 (π/2) | 0.0 – 3.14 | Per-bin rotation budget (radians). Caps worst-case rotation via tanh. |
| **energy_compensate** | False | — | Rescale output RMS to match original. Off by default. |

### n_periods Guide

| Value | Effect |
|-------|--------|
| 1 | Ultra-conservative (global balance only) |
| 2 | Conservative (default, recommended) |
| 3 | Balanced (moderate diversity) |
| 4 | Aggressive (more diversity, higher risk) |

The cutoff adapts to image resolution automatically: `freq_cutoff = n_periods / max(H, W)`. This means `n_periods=2` always affects the same composition-scale frequencies regardless of whether you're generating 512x512 or 2048x2048.

### max_rotation Presets

| Value | Effect |
|-------|--------|
| 1.00 | Conservative — less diversity, very safe |
| 1.57 (π/2) | Balanced (default, recommended) |
| 2.00 | Aggressive — more diversity, some risk |
| 0.00 | Disabled — original uncapped behavior |

## DiversityBoost vs Dummy Token

Both aim to restore diversity in distilled models, but they work at fundamentally different levels:

| | DiversityBoost | Dummy Token |
|---|---|---|
| **Mechanism** | Rotate low-frequency phase in frequency domain | Add/modify padding tokens to shift attention context |
| **Target** | Spatial arrangement (composition skeleton) | Global attention bias (indirect) |
| **Composition change** | Direct and controllable (precise rotation angles) | Indirect, random (butterfly effect) |
| **Diversity source** | Each seed's unique noise phase | Random padding token content |
| **Safety** | Amplitude exactly preserved; Butterworth + tanh double-bounded | No mathematical guarantees |
| **Prompt adherence** | Unaffected (operates on spatial structure, not semantics) | May degrade (alters attention distribution) |

In short: DiversityBoost operates precisely on composition-scale phase in the frequency domain, with mathematically bounded safety guarantees. Dummy token injects perturbation at the token level — the effect is indirect and unpredictable.

## Tips

- **Start with defaults** — strength=1.0, n_periods=2, max_rotation=π/2 is a safe baseline
- **Want more diversity?** Raise `n_periods` first (more frequency bins affected), then `max_rotation` (higher per-bin ceiling)
- **Compatible** with other model patches (ComfyUI-LCS color control, ControlNet, etc.) — operates on a different hook

## Tested Models

| Model | Status |
|-------|--------|
| FLUX2.[Klein] 9B | Tested |
| z-image-turbo | Tested |

Feel free to report results with other distilled models.

## License

MIT
