"""ComfyUI-DiversityBoost V3: Restore composition diversity for distilled diffusion models.

HF attenuation (polynomial frequency modulation) + DCT composition push at step 0.
Training-free, single-step, zero model modification.
"""

from comfy_api.latest import ComfyExtension, io
from .core_node import DiversityBoostCoreV3
from .core_legacy_node import DiversityBoostCoreLegacy


class DiversityBoostExtension(ComfyExtension):
    """V3 ComfyExtension providing the DiversityBoost nodes."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DiversityBoostCoreV3, DiversityBoostCoreLegacy]


async def comfy_entrypoint() -> DiversityBoostExtension:
    """V3 async entry point called by ComfyUI on startup."""
    return DiversityBoostExtension()


# V2 backward compatibility
NODE_CLASS_MAPPINGS = {
    "DiversityBoostCore": DiversityBoostCoreLegacy,
    "DiversityBoostCoreV3": DiversityBoostCoreV3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiversityBoostCore": "Diversity Boost",
    "DiversityBoostCoreV3": "Diversity Boost (V3)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "DiversityBoostExtension",
    "comfy_entrypoint",
    "DiversityBoostCoreV3",
    "DiversityBoostCoreLegacy",
]
