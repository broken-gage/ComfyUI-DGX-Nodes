"""
DGX Nodes: CLIP Vision Loader (Unified Memory)

Loads a CLIP vision model from clip_vision/ directly into CUDA memory. This is
the missing loader surface for WAN-style image-to-video workflows while keeping
the implementation self-contained inside this package.
"""

import logging

import comfy.clip_vision
import comfy.model_management
import folder_paths
import torch

from .performance_metrics import node_timer
from .common import (
    cuda_device_input,
    dgx_mode_input,
    ensure_safetensors_file,
    force_assign_core_model_patcher,
    force_text_encoder_devices,
    load_safetensors_state_dict,
    require_cuda_for_dgx_mode,
    storage_backend_input,
)

logger = logging.getLogger(__name__)


def _load_clip_vision_stock(clip_path):
    clip_vision = comfy.clip_vision.load(clip_path)
    if clip_vision is None:
        raise RuntimeError("ERROR: clip vision file is invalid and does not contain a valid vision model.")
    return clip_vision


def _load_clip_vision_direct(clip_path, device="cuda:0", storage_backend="auto"):
    target_device = torch.device(device)
    ensure_safetensors_file(
        clip_path,
        "CLIPVisionLoaderDGX",
        "Use CLIPVisionLoader for other formats.",
    )

    sd, _metadata, backend_used, gds_used = load_safetensors_state_dict(
        clip_path,
        target_device,
        storage_backend=storage_backend,
    )
    logger.info(
        "[DGX] backend=%s gds=%s | CLIP vision tensors on %s | cuda allocated: %.2f GB",
        backend_used,
        gds_used,
        target_device,
        torch.cuda.memory_allocated(target_device) / 1e9,
    )

    prefix = ""
    convert_keys = False
    if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd:
        prefix = "visual."
        convert_keys = True

    with force_text_encoder_devices(target_device), force_assign_core_model_patcher():
        clip_vision = comfy.clip_vision.load_clipvision_from_sd(
            sd,
            prefix=prefix,
            convert_keys=convert_keys,
        )

    if clip_vision is None:
        raise RuntimeError("ERROR: clip vision file is invalid and does not contain a valid vision model.")

    clip_vision.load_device = target_device
    clip_vision.patcher.load_device = target_device
    clip_vision.patcher.offload_device = target_device
    clip_vision.dtype = comfy.model_management.text_encoder_dtype(target_device)
    comfy.model_management.load_models_gpu([clip_vision.patcher], force_full_load=True)
    return clip_vision, backend_used, gds_used


class CLIPVisionLoaderDGX:
    DESCRIPTION = (
        "Loads a CLIP vision model from clip_vision/. With DGX mode enabled, uses "
        "the direct-to-CUDA unified-memory path. With DGX mode disabled, falls "
        "back to the stock ComfyUI CLIP vision loading pipeline."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (
                    folder_paths.get_filename_list("clip_vision"),
                    {"tooltip": "CLIP vision file from ComfyUI's clip_vision directory."},
                ),
                "dgx_mode": dgx_mode_input(),
                "device": cuda_device_input(),
                "storage_backend": storage_backend_input(),
            }
        }

    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip"
    CATEGORY = "DGX Nodes"

    def load_clip(self, clip_name, dgx_mode=True, device="cuda:0", storage_backend="auto"):
        with node_timer(
            logger,
            "CLIPVisionLoaderDGX",
            clip_name=clip_name,
            dgx_mode=bool(dgx_mode),
            device=device,
            storage_backend=storage_backend,
        ) as metrics:
            clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_name)
            if not dgx_mode:
                logger.info("[DGX] DGX mode disabled for CLIP vision load, using stock pipeline.")
                metrics["path"] = "stock"
                metrics["backend_used"] = "stock"
                metrics["gds_used"] = False
                clip_vision = _load_clip_vision_stock(clip_path)
                return (clip_vision,)

            require_cuda_for_dgx_mode("CLIPVisionLoaderDGX")
            metrics["path"] = "dgx"
            clip_vision, backend_used, gds_used = _load_clip_vision_direct(
                clip_path,
                device=device,
                storage_backend=storage_backend,
            )
            metrics["backend_used"] = backend_used
            metrics["gds_used"] = gds_used
            return (clip_vision,)


NODE_CLASS_MAPPINGS = {
    "CLIPVisionLoaderDGX": CLIPVisionLoaderDGX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPVisionLoaderDGX": "CLIP Vision Loader (Unified Memory)",
}
