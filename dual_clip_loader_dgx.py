"""
DGX Nodes: Dual CLIP Loader (Unified Memory)

Loads two text encoders from text_encoders/ into a single CLIP object using the
same DGX direct-to-CUDA path as the single CLIP loader. This targets stock
DualCLIPLoader compatibility, starting with Flux-style dual text encoders.
"""

import logging

import folder_paths

from .performance_metrics import node_timer
from .clip_loader_dgx import _load_clip_direct_from_paths, _load_clip_stock
from .common import cuda_device_list, dgx_mode_input, require_cuda_for_dgx_mode

logger = logging.getLogger(__name__)

DUAL_CLIP_TYPES = [
    "sdxl",
    "sd3",
    "flux",
    "hunyuan_video",
    "hidream",
    "hunyuan_image",
    "hunyuan_video_15",
    "kandinsky5",
    "kandinsky5_image",
    "ltxv",
    "newbie",
    "ace",
]


class DualCLIPLoaderDGX:
    DESCRIPTION = (
        "Loads two text encoders from text_encoders/. With DGX mode enabled, uses "
        "the direct-to-CUDA unified-memory path. With DGX mode disabled, falls "
        "back to the stock ComfyUI dual CLIP loading pipeline."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
                "type": (DUAL_CLIP_TYPES,),
                "dgx_mode": dgx_mode_input(),
                "device": (cuda_device_list(), {"default": "cuda:0"}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "DGX Nodes"

    def load_clip(self, clip_name1, clip_name2, type, dgx_mode=True, device="cuda:0"):
        with node_timer(
            logger,
            "DualCLIPLoaderDGX",
            clip_name1=clip_name1,
            clip_name2=clip_name2,
            clip_type=type,
            dgx_mode=bool(dgx_mode),
            device=device,
        ) as metrics:
            clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
            clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
            clip_paths = [clip_path1, clip_path2]

            if not dgx_mode:
                logger.info("[DGX] DGX mode disabled for dual CLIP load, using stock pipeline.")
                metrics["path"] = "stock"
                clip = _load_clip_stock(clip_paths, clip_type_name=type)
                return (clip,)

            require_cuda_for_dgx_mode("DualCLIPLoaderDGX")
            metrics["path"] = "dgx"
            clip = _load_clip_direct_from_paths(
                clip_paths,
                clip_type_name=type,
                device=device,
                node_name="DualCLIPLoaderDGX",
            )
            return (clip,)


NODE_CLASS_MAPPINGS = {
    "DualCLIPLoaderDGX": DualCLIPLoaderDGX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualCLIPLoaderDGX": "Dual CLIP Loader (Unified Memory)",
}
