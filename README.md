# ComfyUI-DiversityBoost

Restore composition diversity for distilled diffusion models. Training-free, single-step, zero model modification.

> [中文版 README](README_zh.md)

## The Problem

Step-distilled models (FLUX2.[Klein], z-image-turbo, etc.) generate high-quality images in few steps, but suffer from **composition collapse**: different seeds produce nearly identical layouts. A portrait prompt always puts the subject dead-center; a landscape prompt always uses the same horizon line.

Root cause: distillation freezes the spatial distribution of token norms across seeds, locking the model into a single "average" composition regardless of the initial noise.

## The Fix (V3)

DiversityBoost V3 applies a single post-CFG hook with **polynomial frequency modulation** and a **DCT composition push**:

1. **Polynomial frequency modulation** — smooth, continuous attenuation of high-frequency amplitude. Token-grid normalized (resolution-independent). Near-DC frequencies are protected to prevent brightness/color shift.
2. **DCT composition push** — applies a random low-frequency spatial field that redistributes energy across the latent, nudging the model toward different compositions per seed.

The model then freely reconstructs coherent details at subsequent steps, with per-seed noise driving different reconstruction paths.

Zero model modification. Zero training. One node.

## How It Works

1. Convert the model's prediction to raw latent space
2. **Polynomial frequency modulation** — smooth HF attenuation in the frequency domain, token-grid normalized via DiT patch_size. DC is zeroed for diversity; near-DC frequencies are protected to preserve structure.
3. **DCT spatial field** — synthesize a random 4x4 low-frequency field (zero DC, pink/white/blue noise weighted), normalize to unit std, scale by strength
4. **Multiplicative push** — `modulated * (1 + field)`, clamped to prevent dead zones
5. Convert back

Primary effect at step 0. Optional progressive decay (`linear`/`cosine` schedule) extends HF attenuation into early steps for stronger diversity.

## Nodes

Two nodes are provided for backward compatibility:

| Node | Class Type | Description |
|------|-----------|-------------|
| **Diversity Boost (V3)** | `DiversityBoostCoreV3` | Polynomial frequency modulation, token-grid normalized, near-DC protected. Recommended. |
| **Diversity Boost** | `DiversityBoostCore` | Legacy Butterworth LPF (`n_periods` parameter). Kept for old workflows. |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/facok/ComfyUI-DiversityBoost.git
```

No extra dependencies — only requires PyTorch (ships with ComfyUI).

## Usage

```
MODEL -> [Diversity Boost (V3)] -> MODEL -> KSampler
```

1. Add the **Diversity Boost (V3)** node (category: `sampling`)
2. Connect your model to the input, output to KSampler
3. Generate with different seeds — compositions will vary

## Parameters (V3)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **strength** | 2.0 | 0.0 – 2.0 | Composition push amplitude. 0 = cleanup only, 1.0 = moderate, 2.0 = strong. |
| **clamp** | 0.5 | 0.1 – 3.0 | Upper bound for the multiplicative scale factor. Scale is clamped to [0.1, 1+clamp]. |
| **noise_type** | pink | pink / white / blue | Frequency spectrum of random DCT coefficients. Pink boosts low-freq composition modes. |
| **dc_preserve** | 0.0 | 0.0 – 1.0 | DC amplitude preservation (step 0 only). 0 = max diversity. 1 = preserve original brightness. |
| **energy_compensate** | False | — | Rescale output RMS to match original. Off by default. |
| **hf_factor** | 1.0 | 0.0 – 1.0 | High-frequency attenuation strength. 1.0 = full HF zeroing. |
| **lf_factor** | 0.3 | 0.0 – 1.0 | Low-frequency amplification. 1.0 = +50% boost. |
| **transition** | 2.0 | 0.5 – 4.0 | Polynomial transition shape. 0.5 = steep, 1.0 = linear, 2.0 = smooth, 4.0 = very smooth. |
| **schedule** | linear | flat / linear / cosine | Timestep schedule. `flat` = step 0 only (safe for all samplers). `linear`/`cosine` = progressive decay. |

### schedule Guide

| Mode | Effect | Sampler Compatibility |
|------|--------|----------------------|
| **flat** | Frequency modulation + DCT push at step 0 only. Model has all remaining steps to recover. | All samplers (1st and 2nd order) |
| **linear** | HF attenuation decays linearly over first ~3 steps. DCT push still at step 0 only. | 2nd-order samplers (res_2m, heunpp2) |
| **cosine** | HF attenuation decays with cosine curve. Smoother than linear. | 2nd-order samplers (res_2m, heunpp2) |

**Note:** First-order samplers (euler) are sensitive to denoised modification at step 1+. Use `flat` schedule with 1st-order samplers.

### strength Guide

| Value | Effect |
|-------|--------|
| 0.0 | HF cleanup only (no composition push) |
| 0.5 | Subtle composition variation |
| 1.0 | Moderate composition changes |
| 2.0 | Strong composition changes (default) |

### hf_factor Guide

| Value | Effect |
|-------|--------|
| 0.0 | No HF attenuation (cleanup only) |
| 0.5 | Moderate HF attenuation (HF ~50%) |
| 0.7 | Strong HF attenuation (HF ~30%) |
| 1.0 | Full HF zeroing (default) |

## Legacy Node (V2)

The old `Diversity Boost` node (class type `DiversityBoostCore`) is preserved for backward compatibility. It uses the original Butterworth LPF with the `n_periods` parameter.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **strength** | 0.5 | 0.0 – 2.0 | Composition push amplitude |
| **clamp** | 1.0 | 0.1 – 3.0 | Upper bound for scale factor |
| **noise_type** | pink | pink / white / blue | DCT coefficient spectrum |
| **n_periods** | 2 | 1 – 10 | Butterworth cutoff — spatial periods to preserve |
| **dc_preserve** | 0.0 | 0.0 – 1.0 | DC amplitude preservation |
| **energy_compensate** | False | — | Rescale output RMS |

## Tips

- **Start with V3 defaults** — they are tuned for strong diversity with minimal side effects
- **First-order sampler (euler)?** Use `schedule=flat` to avoid incomplete denoising
- **Second-order sampler (res_2m)?** `schedule=linear` works well for progressive HF release
- **Want more diversity?** Raise `strength` or `hf_factor`
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
