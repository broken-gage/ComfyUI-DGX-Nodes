"""
DGX Nodes: VAE Loader (Unified Memory)

Loads a VAE from vae/ directly into CUDA memory. The implementation stays fully
inside this custom node package so the node can be used without patching ComfyUI
core files.
"""

import logging

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import torch

from .performance_metrics import node_timer
from .common import (
    cuda_device_list,
    dgx_mode_input,
    ensure_safetensors_file,
    force_assign_core_model_patcher,
    load_safetensors_state_dict,
    require_cuda_for_dgx_mode,
)

logger = logging.getLogger(__name__)


def _load_vae_stock(vae_path):
    vae_sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def _load_vae_direct(vae_path, device="cuda:0"):
    target_device = torch.device(device)
    ensure_safetensors_file(
        vae_path,
        "VAELoaderDGX",
        "Use VAELoader for other formats or TAE/pixel-space options.",
    )

    vae_sd, metadata = load_safetensors_state_dict(vae_path, target_device)
    logger.info(
        "[DGX] VAE tensors on %s | cuda allocated: %.2f GB",
        target_device,
        torch.cuda.memory_allocated(target_device) / 1e9,
    )

    with force_assign_core_model_patcher():
        vae = comfy.sd.VAE(sd=vae_sd, device=target_device, metadata=metadata)

    vae.first_stage_model.to(device=target_device, dtype=vae.vae_dtype)
    vae.device = target_device
    vae.patcher.load_device = target_device
    vae.patcher.offload_device = target_device
    vae.throw_exception_if_invalid()
    comfy.model_management.load_models_gpu([vae.patcher], force_full_load=True)
    return vae


class VAELoaderDGX:
    DESCRIPTION = (
        "Loads a VAE from vae/. With DGX mode enabled, uses the direct-to-CUDA "
        "unified-memory path. With DGX mode disabled, falls back to the stock "
        "ComfyUI VAE loading pipeline."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "dgx_mode": dgx_mode_input(),
                "device": (cuda_device_list(), {"default": "cuda:0"}),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "DGX Nodes"

    def load_vae(self, vae_name, dgx_mode=True, device="cuda:0"):
        with node_timer(
            logger,
            "VAELoaderDGX",
            vae_name=vae_name,
            dgx_mode=bool(dgx_mode),
            device=device,
        ) as metrics:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            if not dgx_mode:
                logger.info("[DGX] DGX mode disabled for VAE load, using stock pipeline.")
                metrics["path"] = "stock"
                vae = _load_vae_stock(vae_path)
                return (vae,)

            require_cuda_for_dgx_mode("VAELoaderDGX")
            metrics["path"] = "dgx"
            vae = _load_vae_direct(vae_path, device=device)
            return (vae,)


NODE_CLASS_MAPPINGS = {
    "VAELoaderDGX": VAELoaderDGX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAELoaderDGX": "VAE Loader (Unified Memory)",
}
