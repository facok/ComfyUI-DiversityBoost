# ComfyUI-DiversityBoost

Restore composition diversity for distilled diffusion models. Training-free, single-step, zero model modification.

> [中文版 README](README_zh.md)

## The Problem

Step-distilled models (FLUX2.[Klein], z-image-turbo, etc.) generate high-quality images in few steps, but suffer from **composition collapse**: different seeds produce nearly identical layouts. A portrait prompt always puts the subject dead-center; a landscape prompt always uses the same horizon line.

Root cause: distillation freezes the spatial distribution of token norms across seeds, locking the model into a single "average" composition regardless of the initial noise.

## The Fix

DiversityBoost applies two mechanisms at **step 0 only**, via a single post-CFG hook:

1. **HF attenuation** (Butterworth LPF) — erases the high-frequency spatial anchoring from the model's fully-committed prediction, producing a blurry "composition sketch"
2. **DCT composition push** — applies a random low-frequency spatial field that redistributes energy across the latent, nudging the model toward different compositions

The model then freely reconstructs coherent details at subsequent steps, with per-seed noise driving different reconstruction paths.

Zero model modification. Zero training. One node.

## How It Works

1. Convert the model's step-0 prediction to raw latent space
2. **Butterworth LPF** in frequency domain — attenuate high-frequency amplitude (6th-order, elliptical, resolution-independent)
3. **DCT spatial field** — synthesize a random 4×4 low-frequency field (zero DC, pink noise weighted), normalize to unit std, scale by strength
4. **Multiplicative push** — `blurred × (1 + field)`, clamped to prevent dead zones
5. Convert back

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
| **strength** | 0.5 | 0.0 – 2.0 | Composition push amplitude. 0 = cleanup only, 0.5 = moderate, 1.0 = strong. |
| **clamp** | 1.0 | 0.1 – 3.0 | Upper bound for the multiplicative scale factor. Scale is clamped to [0.1, 1+clamp]. Higher = allow stronger push. |
| **noise_type** | pink | pink / white / blue | Frequency spectrum of random DCT coefficients. Pink boosts low-freq composition modes (recommended). |
| **n_periods** | 2 | 1 – 10 | Butterworth cutoff — spatial periods to preserve. Lower = more HF erased = more diversity. |
| **dc_preserve** | 0.0 | 0.0 – 1.0 | DC amplitude preservation. 0 = max diversity (tone varies per seed). 1 = preserve original brightness. |
| **energy_compensate** | False | — | Rescale output RMS to match original. Off by default. |

### n_periods Guide

n_periods sets the Butterworth -3dB point at FFT bin N. Frequencies above this are strongly attenuated.

| Value | Effect |
|-------|--------|
| 1 | Most aggressive — erases nearly all spatial structure including most DCT composition signal |
| 2 | Recommended — clean frequency gap between DCT composition modes (bin ~1.5) and object-scale details (bin ~5+) |
| 3 | Moderate — preserves more mid-frequency detail |
| 4+ | Mild — less HF erased, less diversity |

### strength Guide

| Value | Effect |
|-------|--------|
| 0.0 | HF cleanup only (no composition push) |
| 0.3 | Subtle composition variation |
| 0.5 | Moderate (default, recommended) |
| 1.0 | Strong composition changes |

## Tips

- **Start with defaults** — strength=0.5, n_periods=2, noise_type=pink is a safe baseline
- **Want more diversity?** Raise `strength`. Keep `n_periods=2` — lowering to 1 kills most DCT composition signal
- **Want cleanup only?** Set `strength=0` — pure HF attenuation, no composition push
- **Compatible** with other model patches (ControlNet, etc.) — operates on a different hook

## Tested Models

| Model | Status |
|-------|--------|
| FLUX2.[Klein] 9B | Tested |
| z-image-turbo | Tested |

Feel free to report results with other distilled models.

## License

MIT
