"""DiversityBoost node — HF attenuation + DCT composition push."""

import time

from comfy_api.latest import io

from .core import build_diversity_fn


class DiversityBoostCore(io.ComfyNode):
    """Restore composition diversity for distilled diffusion models.

    Single post-cfg hook at step 0: first attenuates HF amplitude
    (Butterworth LPF), then applies a random low-frequency DCT spatial
    field to the blurred result.  Push runs AFTER cleanup so its signal
    cannot be erased by downstream processing.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DiversityBoostCore",
            display_name="Diversity Boost",
            category="sampling",
            description="Restore composition diversity for distilled models. "
                        "HF attenuation + DCT composition push at step 0.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("strength", default=0.50, min=0.0, max=2.0, step=0.05,
                               tooltip="Composition push amplitude. "
                                       "0 = cleanup only. 0.5 = moderate. 1.0 = strong."),
                io.Float.Input("clamp", default=1.0, min=0.1, max=3.0, step=0.1,
                               tooltip="Safety clamp for DCT field values."),
                io.Combo.Input("noise_type",
                               options=["pink", "white", "blue"],
                               default="pink",
                               tooltip="Frequency spectrum of random DCT coefficients. "
                                       "pink = stronger composition push (recommended)."),
                io.Int.Input("n_periods", default=2, min=1, max=10, step=1,
                             tooltip="Butterworth cutoff. 2 = preserves DCT signal."),
                io.Float.Input("dc_preserve", default=0.0, min=0.0, max=1.0, step=0.1,
                               tooltip="DC amplitude preservation (0 = max diversity)."),
                io.Boolean.Input("energy_compensate", default=False,
                                 tooltip="Rescale output energy to match original."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return time.time()

    @classmethod
    def execute(cls, model, strength, clamp, noise_type,
                n_periods, dc_preserve, energy_compensate) -> io.NodeOutput:
        m = model.clone()

        m.set_model_sampler_post_cfg_function(
            build_diversity_fn(
                strength=strength,
                clamp_val=clamp,
                noise_type=noise_type,
                n_periods=n_periods,
                dc_preserve=dc_preserve,
                energy_compensate=energy_compensate,
            ),
        )

        return io.NodeOutput(m)
