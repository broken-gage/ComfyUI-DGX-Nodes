"""
DGX Nodes: Upscale Model Loader (Unified Memory)

Loads an upscaler model from upscale_models/. Safetensors files can use the DGX
backend stack; unsupported formats automatically fall back to the stock ComfyUI
upscale-model loading pipeline.
"""

import logging

import comfy.utils
import folder_paths
import torch
from spandrel import ImageModelDescriptor, ModelLoader

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY

    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
except Exception:
    pass

from .common import (
    cuda_device_input,
    dgx_mode_input,
    is_safetensors_file,
    load_safetensors_state_dict,
    require_cuda_for_dgx_mode,
    storage_backend_input,
)
from .performance_metrics import node_timer

logger = logging.getLogger(__name__)


def _normalize_upscale_state_dict(sd):
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    return sd


def _load_upscale_model_stock(model_path):
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    sd = _normalize_upscale_state_dict(sd)
    out = ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(out, ImageModelDescriptor):
        raise RuntimeError("Upscale model must be a single-image model.")
    return out


def _load_upscale_model_direct(model_path, device="cuda:0", storage_backend="auto"):
    target_device = torch.device(device)
    sd, _metadata, backend_used, gds_used = load_safetensors_state_dict(
        model_path,
        target_device,
        storage_backend=storage_backend,
    )
    sd = _normalize_upscale_state_dict(sd)
    out = ModelLoader(device=target_device).load_from_state_dict(sd).eval()
    if not isinstance(out, ImageModelDescriptor):
        raise RuntimeError("Upscale model must be a single-image model.")
    return out, backend_used, gds_used


class UpscaleModelLoaderDGX:
    DESCRIPTION = (
        "Loads an upscaler model from upscale_models/. With DGX mode enabled, "
        "safetensors models use the DGX backend stack. Unsupported file formats "
        "such as .pth automatically fall back to the stock ComfyUI upscaler "
        "loader instead of failing."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("upscale_models"),
                    {"tooltip": "Upscaler model file from ComfyUI's upscale_models directory."},
                ),
                "dgx_mode": dgx_mode_input(),
                "device": cuda_device_input(),
                "storage_backend": storage_backend_input(),
            }
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "DGX Nodes"

    def load_model(self, model_name, dgx_mode=True, device="cuda:0", storage_backend="auto"):
        with node_timer(
            logger,
            "UpscaleModelLoaderDGX",
            model_name=model_name,
            dgx_mode=bool(dgx_mode),
            device=device,
            storage_backend=storage_backend,
        ) as metrics:
            model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)

            if not dgx_mode:
                logger.info("[DGX] DGX mode disabled for upscale model load, using stock pipeline.")
                metrics["path"] = "stock"
                metrics["backend_used"] = "stock"
                metrics["gds_used"] = False
                return (_load_upscale_model_stock(model_path),)

            if not is_safetensors_file(model_path):
                logger.info(
                    "[DGX] Upscale model format is not supported by the DGX direct-load path, "
                    "falling back to stock pipeline: %s",
                    model_name,
                )
                metrics["path"] = "stock"
                metrics["backend_used"] = "stock"
                metrics["gds_used"] = False
                metrics["format_fallback"] = True
                return (_load_upscale_model_stock(model_path),)

            require_cuda_for_dgx_mode("UpscaleModelLoaderDGX")
            metrics["path"] = "dgx"
            model, backend_used, gds_used = _load_upscale_model_direct(
                model_path,
                device=device,
                storage_backend=storage_backend,
            )
            metrics["backend_used"] = backend_used
            metrics["gds_used"] = gds_used
            metrics["format_fallback"] = False
            return (model,)


NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoaderDGX": UpscaleModelLoaderDGX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleModelLoaderDGX": "Upscale Model Loader (Unified Memory)",
}
