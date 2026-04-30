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
    cuda_device_input,
    dgx_mode_input,
    ensure_safetensors_file,
    force_assign_core_model_patcher,
    load_safetensors_state_dict,
    mark_patcher_as_loaded,
    require_cuda_for_dgx_mode,
    storage_backend_input,
)

logger = logging.getLogger(__name__)


def _load_vae_stock(vae_path):
    vae_sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def _load_vae_direct(vae_path, device="cuda:0", storage_backend="auto"):
    target_device = torch.device(device)
    ensure_safetensors_file(
        vae_path,
        "VAELoaderDGX",
        "Use VAELoader for other formats or TAE/pixel-space options.",
    )

    vae_sd, metadata, backend_used, gds_used = load_safetensors_state_dict(
        vae_path,
        target_device,
        storage_backend=storage_backend,
    )
    logger.info(
        "[DGX] backend=%s gds=%s | VAE tensors on %s | cuda allocated: %.2f GB",
        backend_used,
        gds_used,
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
    # Correct model_management tracking before load_models_gpu: assign=True bypasses
    # ModelPatcher.load() so model_loaded_weight_memory stays 0 — without this,
    # load_models_gpu would call free_memory() and unnecessarily evict other models.
    mark_patcher_as_loaded(vae.patcher, target_device)
    comfy.model_management.load_models_gpu([vae.patcher], force_full_load=True)
    return vae, backend_used, gds_used


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
                "vae_name": (
                    folder_paths.get_filename_list("vae"),
                    {"tooltip": "VAE file from ComfyUI's vae directory."},
                ),
                "dgx_mode": dgx_mode_input(),
                "device": cuda_device_input(),
                "storage_backend": storage_backend_input(),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "DGX Nodes"

    def load_vae(self, vae_name, dgx_mode=True, device="cuda:0", storage_backend="auto"):
        with node_timer(
            logger,
            "VAELoaderDGX",
            vae_name=vae_name,
            dgx_mode=bool(dgx_mode),
            device=device,
            storage_backend=storage_backend,
        ) as metrics:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            if not dgx_mode:
                logger.info("[DGX] DGX mode disabled for VAE load, using stock pipeline.")
                metrics["path"] = "stock"
                metrics["backend_used"] = "stock"
                metrics["gds_used"] = False
                vae = _load_vae_stock(vae_path)
                return (vae,)

            require_cuda_for_dgx_mode("VAELoaderDGX")
            metrics["path"] = "dgx"
            vae, backend_used, gds_used = _load_vae_direct(
                vae_path,
                device=device,
                storage_backend=storage_backend,
            )
            metrics["backend_used"] = backend_used
            metrics["gds_used"] = gds_used
            return (vae,)


NODE_CLASS_MAPPINGS = {
    "VAELoaderDGX": VAELoaderDGX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAELoaderDGX": "VAE Loader (Unified Memory)",
}
