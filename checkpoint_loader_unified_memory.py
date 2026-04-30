"""
DGX Nodes: Checkpoint Loader (Unified Memory)

Loads a safetensors checkpoint directly into CUDA memory, bypassing the standard
CPU staging buffer. Designed for NVIDIA Grace-Blackwell DGX Spark where CPU and
GPU share the same physical memory pool (unified memory, no GDS).

Root cause of the standard loader wasting memory on unified memory:
  - model_management.unet_inital_load_device() checks mem_dev > mem_cpu (strict).
    On GB10 both are ~122 GB (same pool), so mem_dev == mem_cpu -> returns CPU.
  - CoreModelPatcher is aliased to base ModelPatcher when aimdo.so is absent,
    so is_dynamic() returns False -> load_model_weights uses assign=False -> CUDA
    tensors get copied into CPU model parameters and freed.
  - Net result: disk -> CUDA (3.5 GB) -> copy to CPU (3.5 GB) -> CUDA freed.
    Model ends on CPU; GPU has to re-load it at inference time.

This node forces:
  - model placed on CUDA directly (bypasses unet_inital_load_device)
  - assign=True in load_model_weights (sd CUDA tensors become model params, no copy)
  - load_models_gpu called so ComfyUI tracks the model as GPU-resident

Peak memory is ~2x model size briefly during get_model + assign
(skeleton CUDA + sd CUDA), but after loading only 1x remains on CUDA.
The remaining CLIP and VAE keys are also kept on CUDA and are constructed
through the same unified-memory-aware path inside this node package.
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
    force_assign_core_model_patcher,
    gpu_text_encoder_model_options,
    load_safetensors_state_dict,
    mark_patcher_as_loaded,
    normalize_clip_metadata_tensors,
    require_cuda_for_dgx_mode,
    storage_backend_input,
)

logger = logging.getLogger(__name__)


def _load_checkpoint_stock(ckpt_path):
    out = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    return out[:3]


class CheckpointLoaderUnifiedMemory:
    DESCRIPTION = (
        "Loads a checkpoint and produces the same MODEL / CLIP / VAE outputs as "
        "CheckpointLoaderSimple. With DGX mode enabled, uses the direct-to-CUDA "
        "unified-memory path for DGX Spark systems. With DGX mode disabled, falls "
        "back to the stock ComfyUI loading pipeline."
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "Checkpoint file from ComfyUI's checkpoints directory."},
                ),
                "dgx_mode": dgx_mode_input(),
                "device": cuda_device_input(),
                "storage_backend": storage_backend_input(),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "DGX Nodes"

    def load_checkpoint(self, ckpt_name, dgx_mode=True, device="cuda:0", storage_backend="auto"):
        with node_timer(
            logger,
            "CheckpointLoaderUnifiedMemory",
            checkpoint=ckpt_name,
            dgx_mode=bool(dgx_mode),
            device=device,
            storage_backend=storage_backend,
        ) as metrics:
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            if not dgx_mode:
                logger.info("[DGX] DGX mode disabled for checkpoint load, using stock pipeline.")
                metrics["path"] = "stock"
                metrics["backend_used"] = "stock"
                metrics["gds_used"] = False
                model, clip, vae = _load_checkpoint_stock(ckpt_path)
                metrics["has_clip"] = clip is not None
                metrics["has_vae"] = vae is not None
                return (model, clip, vae)

            require_cuda_for_dgx_mode("CheckpointLoaderUnifiedMemory")
            ensure_safetensors_file(
                ckpt_path,
                "CheckpointLoaderUnifiedMemory",
                "Use CheckpointLoaderSimple for .ckpt/.pt files.",
            )

            target_device = torch.device(device)
            load_device = target_device
            offload_device = target_device

            sd, metadata, backend_used, gds_used = load_safetensors_state_dict(
                ckpt_path,
                target_device,
                storage_backend=storage_backend,
            )
            metrics["path"] = "dgx"
            metrics["backend_used"] = backend_used
            metrics["gds_used"] = gds_used
            metrics["tensors"] = len(sd)

            logger.info(
                "[DGX] backend=%s gds=%s | %d tensors on %s | cuda allocated: %.2f GB",
                backend_used,
                gds_used,
                len(sd),
                target_device,
                torch.cuda.memory_allocated(target_device) / 1e9,
            )

            prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
            parameters = comfy.utils.calculate_parameters(sd, prefix)
            weight_dtype = comfy.utils.weight_dtype(sd, prefix)

            sd, metadata = comfy.utils.convert_old_quants(sd, prefix, metadata=metadata)

            model_config = comfy.model_detection.model_config_from_unet(sd, prefix, metadata=metadata)
            if model_config is None:
                raise RuntimeError(
                    f"[DGX] Could not detect model type for: {ckpt_path}\n"
                    "Ensure the file is a valid SD/SDXL/SD3/Flux checkpoint."
                )

            unet_weight_dtype = list(model_config.supported_inference_dtypes)
            eff_weight_dtype = None if model_config.quant_config is not None else weight_dtype
            unet_dtype = comfy.model_management.unet_dtype(
                model_params=parameters,
                supported_dtypes=unet_weight_dtype,
                weight_dtype=eff_weight_dtype,
            )
            manual_cast_dtype = comfy.model_management.unet_manual_cast(
                None if model_config.quant_config is not None else unet_dtype,
                load_device,
                model_config.supported_inference_dtypes,
            )
            model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

            # Skeleton on target_device: assign=True promotes the already-CUDA sd tensors
            # directly into model parameters (no CPU staging copy).
            # meta device is incompatible with comfy_cast_weights ops layers — kept on
            # target_device to avoid NotImplementedError during inference.
            model = model_config.get_model(sd, prefix, device=target_device)
            model_patcher = comfy.model_patcher.ModelPatcher(
                model,
                load_device=load_device,
                offload_device=offload_device,
            )

            # Move any metadata tensors (comfy_quant, spiece_model, etc.) to CPU before
            # load_model_weights; ops.py reads them via .numpy() which requires CPU.
            sd = normalize_clip_metadata_tensors(sd)
            model.load_model_weights(sd, prefix, assign=True)

            dm_devices = set(p.device.type for p in model.diffusion_model.parameters())
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

            scaled_fp8_list = [
                key[: -len("scaled_fp8")]
                for key in list(sd.keys())
                if key.endswith(".scaled_fp8")
            ]
            if scaled_fp8_list:
                out_sd = {
                    key: value
                    for key, value in sd.items()
                    if not any(key.startswith(prefix_value) for prefix_value in scaled_fp8_list)
                }
                for prefix_value in scaled_fp8_list:
                    quant_sd, _ = comfy.utils.convert_old_quants(sd, prefix_value, metadata={})
                    out_sd.update(quant_sd)
                sd = out_sd

            vae = None
            vae_sd = comfy.utils.state_dict_prefix_replace(
                sd, {key: "" for key in model_config.vae_key_prefix}, filter_keys=True
            )
            vae_sd = model_config.process_vae_state_dict(vae_sd)
            if len(vae_sd) > 0:
                with force_assign_core_model_patcher():
                    vae = comfy.sd.VAE(sd=vae_sd, metadata=metadata, device=target_device)
                vae.first_stage_model.to(device=target_device, dtype=vae.vae_dtype)
                vae.device = target_device
                vae.patcher.load_device = target_device
                vae.patcher.offload_device = target_device
                vae.throw_exception_if_invalid()
                mark_patcher_as_loaded(vae.patcher, target_device)
                comfy.model_management.load_models_gpu([vae.patcher], force_full_load=True)

            clip = None
            clip_target = model_config.clip_target(state_dict=sd)
            if clip_target is not None:
                clip_sd = model_config.process_clip_state_dict(sd)
                if clip_sd:
                    clip_sd = normalize_clip_metadata_tensors(clip_sd)
                    clip_params = comfy.utils.calculate_parameters(clip_sd)
                    with force_assign_core_model_patcher():
                        # gpu_text_encoder_model_options uses initial_device=meta so the
                        # CLIP skeleton has zero physical footprint and CLIP.__init__ does
                        # NOT call load_models_gpu internally (meta != load_device).
                        clip = comfy.sd.CLIP(
                            clip_target,
                            embedding_directory=folder_paths.get_folder_paths("embeddings"),
                            tokenizer_data=clip_sd,
                            parameters=clip_params,
                            state_dict=clip_sd,
                            model_options=gpu_text_encoder_model_options(target_device),
                        )
                    mark_patcher_as_loaded(clip.patcher, target_device)
                    comfy.model_management.load_models_gpu([clip.patcher], force_full_load=True)
                else:
                    logger.warning("[DGX] No CLIP/text encoder weights found in checkpoint.")

            metrics["has_clip"] = clip is not None
            metrics["has_vae"] = vae is not None
            logger.info("[DGX] Load complete")
            return (model_patcher, clip, vae)


NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderUnifiedMemory": CheckpointLoaderUnifiedMemory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoaderUnifiedMemory": "Checkpoint Loader (Unified Memory)",
}
