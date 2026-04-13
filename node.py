"""DiversityBoost node — ComfyUI node for frequency-domain composition diversity."""

import time

from comfy_api.latest import io

from .phase_inject import build_phase_injection_fn


class DiversityBoost(io.ComfyNode):
    """Restore composition diversity lost during distillation.

    Injects seed-dependent low-frequency phase from initial noise into model
    prediction, while attenuating high-frequency amplitude to prevent
    'frequency shearing' (limb deformity from spatially incoherent LF/HF).
    Operates at step 0 only via post-CFG hook.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DiversityBoost",
            display_name="Diversity Boost",
            category="sampling",
            description="Restore composition diversity for distilled models. "
                        "Injects seed-dependent low-frequency phase from initial "
                        "noise into model prediction. Different seeds produce "
                        "different compositions instead of identical layouts. "
                        "Step 0 only, zero risk to model internals.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("strength", default=1.00, min=0.0, max=1.0, step=0.05,
                               tooltip="Phase rotation strength (gamma). "
                                       "0 = no effect. 1 = full rotation toward noise phase. "
                                       "With max_rotation cap, 1.0 is safe — "
                                       "the cap prevents any bin from over-rotating. "
                                       "To increase diversity further, raise freq_cutoff "
                                       "(more bins) or max_rotation (higher ceiling)."),
                io.Float.Input("freq_cutoff", default=0.02, min=0.005, max=0.05, step=0.005,
                               tooltip="Butterworth LPF cutoff in normalized frequency. "
                                       "Frequencies below cutoff are modified, "
                                       "above are protected (steep 6th-order rolloff). "
                                       "Controls how many spatial periods are affected: "
                                       "0.010 = ≤2 periods (global balance only), "
                                       "0.015 = ≤3 periods (conservative), "
                                       "0.020 = ≤4 periods (balanced, recommended), "
                                       "0.025 = ≤5 periods (aggressive, higher risk)."),
                io.Float.Input("max_rotation", default=1.57, min=0.0, max=3.14, step=0.05,
                               tooltip="Per-bin rotation budget in radians (0 = no cap). "
                                       "Caps the maximum phase rotation at any single "
                                       "frequency bin via tanh soft saturation. "
                                       "Decouples diversity strength from tail risk. "
                                       "Default π/2 ≈ 1.57 is the natural scale "
                                       "(mean |θ| of the noise prior). At this value: "
                                       "composition bins (e.g. bin (1,0)) max 87°, "
                                       "transition bins (e.g. bin (0,3)) max 68°, "
                                       "object bins (e.g. bin (8,0)) max 2°. "
                                       "1.00 = conservative (less diversity, very safe), "
                                       "1.57 = balanced (π/2, recommended), "
                                       "2.00 = aggressive (more diversity, some risk), "
                                       "0.00 = disabled (original uncapped behavior)."),
                io.Float.Input("hf_preserve", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="High-frequency amplitude preservation (0-1). "
                                       "Controls how much high-freq detail survives "
                                       "after phase injection. "
                                       "Distilled models produce fully-formed predictions "
                                       "at step 0 — rotating low-freq phase while keeping "
                                       "high-freq detail causes 'frequency shearing': "
                                       "the body moves but fingers/edges stay pinned at "
                                       "original positions, causing limb deformity. "
                                       "0.0 = full blur (like teacher step 0, safest), "
                                       "0.3 = keep 30% HF (some semantic hints preserved), "
                                       "1.0 = keep all HF (risks shearing)."),
                io.Boolean.Input("energy_compensate", default=False,
                                 tooltip="Rescale output RMS to match original prediction. "
                                         "When hf_preserve < 1, high-freq amplitude is "
                                         "attenuated, reducing total energy. Enable this "
                                         "to compensate by scaling the result back to the "
                                         "original energy level. Off by default."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return time.time()

    @classmethod
    def execute(cls, model, strength, freq_cutoff, max_rotation, hf_preserve, energy_compensate) -> io.NodeOutput:
        m = model.clone()

        if strength > 1e-6:
            m.set_model_sampler_post_cfg_function(
                build_phase_injection_fn(
                    strength=strength,
                    freq_cutoff=freq_cutoff,
                    max_rotation=max_rotation,
                    hf_preserve=hf_preserve,
                    energy_compensate=energy_compensate,
                ),
            )

        return io.NodeOutput(m)
