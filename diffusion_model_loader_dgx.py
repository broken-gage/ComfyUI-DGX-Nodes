"""
DGX Nodes: UNET Loader (Unified Memory)

Loads a standalone diffusion model (UNet/transformer) from diffusion_models/
directly into CUDA memory, bypassing the CPU staging buffer.

Same approach as CheckpointLoaderUnifiedMemory:
  - safetensors.safe_open(device="cuda") -> one allocation, no CPU buffer
  - get_model(device=target_device) -> skeleton created on CUDA
  - load_model_weights(assign=True) -> sd CUDA tensors become model params, no copy
  - load_models_gpu -> registers model as GPU-resident in ComfyUI

Mirrors load_diffusion_model_state_dict from comfy/sd.py, but forces CUDA.
"""

import logging

import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils
import folder_paths
import torch

from .performance_metrics import node_timer
from .common import (
    cuda_device_input,
    dgx_mode_input,
    ensure_safetensors_file,
    load_safetensors_state_dict,
    mark_patcher_as_loaded,
    normalize_clip_metadata_tensors,
    require_cuda_for_dgx_mode,
    storage_backend_input,
)

logger = logging.getLogger(__name__)


def _model_options_from_weight_dtype(weight_dtype):
    model_options = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2
    return model_options


def _load_unet_stock(unet_path, weight_dtype="default"):
    return comfy.sd.load_diffusion_model(
        unet_path,
        model_options=_model_options_from_weight_dtype(weight_dtype),
    )


def _load_unet_direct(
    unet_path,
    weight_dtype="default",
    device="cuda:0",
    load_threads=1,
    storage_backend="auto",
):
    target_device = torch.device(device)
    model_options = _model_options_from_weight_dtype(weight_dtype)

    ensure_safetensors_file(
        unet_path,
        "UNETLoaderDGX",
        "Use UNETLoader for other formats.",
    )

    sd, metadata, backend_used, gds_used = load_safetensors_state_dict(
        unet_path,
        target_device,
        load_threads=load_threads,
        storage_backend=storage_backend,
    )
    logger.info(
        "[DGX] backend=%s gds=%s | %d tensors on %s | cuda allocated: %.2f GB",
        backend_used,
        gds_used,
        len(sd),
        target_device,
        torch.cuda.memory_allocated(target_device) / 1e9,
    )

    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(
        sd,
        {diffusion_model_prefix: ""},
        filter_keys=True,
    )
    if len(temp_sd) > 0:
        sd = temp_sd

    custom_operations = model_options.get("custom_operations")
    if custom_operations is None:
        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

    model_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
    if model_config is not None:
        new_sd = sd
    else:
        new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:
            model_config = comfy.model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                raise RuntimeError(f"[DGX] Could not detect model type for: {unet_path}")
        else:
            model_config = comfy.model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                raise RuntimeError(f"[DGX] Could not detect model type for: {unet_path}")

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)
            new_sd = {}
            for key in diffusers_keys:
                if key in sd:
                    new_sd[diffusers_keys[key]] = sd.pop(key)
                else:
                    logger.warning("%s %s", diffusers_keys[key], key)

    parameters = comfy.utils.calculate_parameters(new_sd)
    detected_weight_dtype = comfy.utils.weight_dtype(new_sd)
    load_device = target_device
    offload_device = target_device

    if model_config.quant_config is not None:
        detected_weight_dtype = None

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    unet_dtype = model_options.get("dtype")
    if unet_dtype is None:
        unet_dtype = comfy.model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=unet_weight_dtype,
            weight_dtype=detected_weight_dtype,
        )

    if model_config.quant_config is not None:
        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            None,
            load_device,
            model_config.supported_inference_dtypes,
        )
    else:
        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            unet_dtype,
            load_device,
            model_config.supported_inference_dtypes,
        )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if custom_operations is not None:
        model_config.custom_operations = custom_operations

    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    # Skeleton on target_device: assign=True promotes the already-CUDA sd tensors
    # directly into model parameters (no CPU staging copy).
    # meta device was tried but is incompatible with comfy_cast_weights ops layers
    # (manual_cast.Linear) which call cast_bias_weight -> cast_to -> r.copy_(weight)
    # on the still-meta weight, causing NotImplementedError at inference time.
    model = model_config.get_model(new_sd, "", device=target_device)
    model_patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
    )
    # Move any metadata tensors (comfy_quant, spiece_model, etc.) to CPU before
    # load_model_weights; ops.py reads them via .numpy() which requires CPU.
    new_sd = normalize_clip_metadata_tensors(new_sd)
    model.load_model_weights(new_sd, "", assign=True)

    dm_devices = set(param.device.type for param in model.diffusion_model.parameters())
    logger.info(
        "[DGX] Diffusion model params on: %s | cuda allocated: %.2f GB",
        dm_devices,
        torch.cuda.memory_allocated(target_device) / 1e9,
    )

    # Correct model_management tracking before load_models_gpu: assign=True bypasses
    # ModelPatcher.load() so model_loaded_weight_memory stays 0 — without this,
    # load_models_gpu would call free_memory() and unnecessarily evict other models.
    mark_patcher_as_loaded(model_patcher, target_device)
    comfy.model_management.load_models_gpu([model_patcher], force_full_load=True)
    model_patcher.cached_patcher_init = (
        _load_unet_model_only_direct,
        (unet_path, weight_dtype, device, load_threads, storage_backend),
    )
    return model_patcher, backend_used, gds_used


def _load_unet_model_only_direct(
    unet_path,
    weight_dtype="default",
    device="cuda:0",
    load_threads=1,
    storage_backend="auto",
):
    model, _backend_used, _gds_used = _load_unet_direct(
        unet_path,
        weight_dtype=weight_dtype,
        device=device,
        load_threads=load_threads,
        storage_backend=storage_backend,
    )
    return model


class UNETLoaderDGX:
    DESCRIPTION = (
        "Loads a standalone diffusion model from diffusion_models/. With DGX mode "
        "enabled, uses the direct-to-CUDA unified-memory path. With DGX mode "
        "disabled, falls back to the stock ComfyUI UNET loading pipeline. Also "
        "returns the dgx_mode state as a boolean output for downstream workflow "
        "toggle wiring."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "Diffusion model / UNET file from ComfyUI's diffusion_models directory."},
                ),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {
                        "advanced": True,
                        "tooltip": "Optional inference dtype override for the loaded diffusion model.",
                    },
                ),
                "dgx_mode": dgx_mode_input(),
                "device": cuda_device_input(),
                "load_threads": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "Parallel loader threads used by the plain safetensors backend.",
                    },
                ),
                "storage_backend": storage_backend_input(),
            }
        }

    RETURN_TYPES = ("MODEL", "BOOLEAN")
    RETURN_NAMES = ("model", "dgx_mode")
    FUNCTION = "load_unet"
    CATEGORY = "DGX Nodes"

    def load_unet(
        self,
        unet_name,
        weight_dtype,
        dgx_mode=True,
        device="cuda:0",
        load_threads=1,
        storage_backend="auto",
    ):
        with node_timer(
            logger,
            "UNETLoaderDGX",
            unet_name=unet_name,
            weight_dtype=weight_dtype,
            dgx_mode=bool(dgx_mode),
            device=device,
            load_threads=int(load_threads),
            storage_backend=storage_backend,
        ) as metrics:
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            if not dgx_mode:
                logger.info("[DGX] DGX mode disabled for UNET load, using stock pipeline.")
                metrics["path"] = "stock"
                metrics["backend_used"] = "stock"
                metrics["gds_used"] = False
                model = _load_unet_stock(unet_path, weight_dtype=weight_dtype)
                return (model, bool(dgx_mode))

            require_cuda_for_dgx_mode("UNETLoaderDGX")
            metrics["path"] = "dgx"
            model, backend_used, gds_used = _load_unet_direct(
                unet_path,
                weight_dtype=weight_dtype,
                device=device,
                load_threads=load_threads,
                storage_backend=storage_backend,
            )
            metrics["backend_used"] = backend_used
            metrics["gds_used"] = gds_used
            return (model, bool(dgx_mode))


DiffusionModelLoaderDGX = UNETLoaderDGX


NODE_CLASS_MAPPINGS = {
    "UNETLoaderDGX": UNETLoaderDGX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UNETLoaderDGX": "UNET Loader (Unified Memory)",
}
