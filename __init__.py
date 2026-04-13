"""ComfyUI-DiversityBoost: Restore composition diversity for distilled diffusion models.

Injects seed-dependent low-frequency phase from initial noise into the model's
denoised prediction, making different seeds produce different compositions instead
of identical layouts.  Training-free, single-step (step 0 only), zero model modification.
"""

# V3 ComfyExtension entry point
from comfy_api.latest import ComfyExtension, io
from .node import DiversityBoost


class DiversityBoostExtension(ComfyExtension):
    """V3 ComfyExtension providing the DiversityBoost node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DiversityBoost]


async def comfy_entrypoint() -> DiversityBoostExtension:
    """V3 async entry point called by ComfyUI on startup."""
    return DiversityBoostExtension()


# V2 backward compatibility
NODE_CLASS_MAPPINGS = {
    "DiversityBoost": DiversityBoost,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiversityBoost": "Diversity Boost",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "DiversityBoostExtension",
    "comfy_entrypoint",
]
