"""
DGX Nodes: CLIP Loader (Unified Memory)

Loads a text encoder from text_encoders/ directly into CUDA memory. This keeps
the entire load path inside the DGX node package and avoids requiring changes to
ComfyUI core files.
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
    gpu_text_encoder_model_options,
    load_safetensors_state_dict,
    normalize_clip_metadata_tensors,
    require_cuda_for_dgx_mode,
    storage_backend_input,
)

logger = logging.getLogger(__name__)

CLIP_TYPES = [
    "stable_diffusion",
    "stable_cascade",
    "sd3",
    "stable_audio",
    "mochi",
    "ltxv",
    "pixart",
    "cosmos",
    "lumina2",
    "wan",
    "hidream",
    "chroma",
    "ace",
    "omnigen2",
    "qwen_image",
    "hunyuan_image",
    "flux2",
    "ovis",
    "longcat_image",
]


def _clip_type_from_name(clip_type_name):
    clip_type = getattr(comfy.sd.CLIPType, clip_type_name.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
    return clip_type


def _load_clip_stock(clip_paths, clip_type_name="stable_diffusion"):
    return comfy.sd.load_clip(
        ckpt_paths=clip_paths,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=_clip_type_from_name(clip_type_name),
        model_options={},
    )


def _load_clip_direct_from_paths(
    clip_paths,
    clip_type_name="stable_diffusion",
    device="cuda:0",
    node_name="CLIPLoaderDGX",
    storage_backend="auto",
):
    target_device = torch.device(device)
    state_dicts = []
    backend_used = None
    gds_used = False
    for clip_path in clip_paths:
        ensure_safetensors_file(
            clip_path,
            node_name,
            "Use CLIPLoader for other formats.",
        )
        clip_sd, metadata, clip_backend_used, clip_gds_used = load_safetensors_state_dict(
            clip_path,
            target_device,
            storage_backend=storage_backend,
        )
        clip_sd, metadata = comfy.utils.convert_old_quants(clip_sd, model_prefix="", metadata=metadata)
        state_dicts.append(normalize_clip_metadata_tensors(clip_sd))
        backend_used = clip_backend_used
        gds_used = gds_used or clip_gds_used

    model_options = gpu_text_encoder_model_options(target_device)

    logger.info(
        "[DGX] backend=%s gds=%s | CLIP tensors on %s | cuda allocated: %.2f GB",
        backend_used,
        gds_used,
        target_device,
        torch.cuda.memory_allocated(target_device) / 1e9,
    )

    with force_assign_core_model_patcher():
        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=state_dicts,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=_clip_type_from_name(clip_type_name),
            model_options=model_options,
        )

    comfy.model_management.load_models_gpu([clip.patcher], force_full_load=True)
    clip.patcher.cached_patcher_init = (
        _load_clip_model_patcher_direct,
        (tuple(clip_paths), clip_type_name, device, storage_backend),
    )
    return clip, backend_used, gds_used


def _load_clip_direct(clip_path, clip_type_name="stable_diffusion", device="cuda:0", storage_backend="auto"):
    return _load_clip_direct_from_paths(
        [clip_path],
        clip_type_name=clip_type_name,
        device=device,
        node_name="CLIPLoaderDGX",
        storage_backend=storage_backend,
    )


def _load_clip_model_patcher_direct(
    clip_paths,
    clip_type_name="stable_diffusion",
    device="cuda:0",
    storage_backend="auto",
):
    clip, _backend_used, _gds_used = _load_clip_direct_from_paths(
        list(clip_paths),
        clip_type_name=clip_type_name,
        device=device,
        storage_backend=storage_backend,
    )
    return clip.patcher


class CLIPLoaderDGX:
    DESCRIPTION = (
        "Loads a text encoder from text_encoders/. With DGX mode enabled, uses "
        "the direct-to-CUDA unified-memory path. With DGX mode disabled, falls "
        "back to the stock ComfyUI CLIP loading pipeline."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (
                    folder_paths.get_filename_list("text_encoders"),
                    {"tooltip": "Text encoder file from ComfyUI's text_encoders directory."},
                ),
                "type": (CLIP_TYPES, {"tooltip": "Target CLIP family / model type used to construct the text encoder."}),
                "dgx_mode": dgx_mode_input(),
                "device": cuda_device_input(),
                "storage_backend": storage_backend_input(),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "DGX Nodes"

    def load_clip(self, clip_name, type="stable_diffusion", dgx_mode=True, device="cuda:0", storage_backend="auto"):
        with node_timer(
            logger,
            "CLIPLoaderDGX",
            clip_name=clip_name,
            clip_type=type,
            dgx_mode=bool(dgx_mode),
            device=device,
            storage_backend=storage_backend,
        ) as metrics:
            clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
            if not dgx_mode:
                logger.info("[DGX] DGX mode disabled for CLIP load, using stock pipeline.")
                metrics["path"] = "stock"
                metrics["backend_used"] = "stock"
                metrics["gds_used"] = False
                clip = _load_clip_stock([clip_path], clip_type_name=type)
                return (clip,)

            require_cuda_for_dgx_mode("CLIPLoaderDGX")
            metrics["path"] = "dgx"
            clip, backend_used, gds_used = _load_clip_direct(
                clip_path,
                clip_type_name=type,
                device=device,
                storage_backend=storage_backend,
            )
            metrics["backend_used"] = backend_used
            metrics["gds_used"] = gds_used
            return (clip,)


NODE_CLASS_MAPPINGS = {
    "CLIPLoaderDGX": CLIPLoaderDGX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPLoaderDGX": "CLIP Loader (Unified Memory)",
}
