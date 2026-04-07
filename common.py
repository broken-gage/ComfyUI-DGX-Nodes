import contextlib
from concurrent.futures import ThreadPoolExecutor

import comfy.model_management
import comfy.model_patcher
import safetensors
import torch

DGX_MODE_TOOLTIP = (
    "ON: use the DGX unified-memory direct-to-CUDA loading path. "
    "OFF: fall back to the stock ComfyUI loading pipeline."
)

_CPU_ONLY_CLIP_SUFFIXES = (
    "comfy_quant",
    "spiece_model",
    "gemma_spiece_model",
    "jina_spiece_model",
    "tekken_model",
    "mt5xl.spiece_model",
)


def cuda_device_list():
    try:
        count = torch.cuda.device_count()
        return [f"cuda:{index}" for index in range(count)] if count > 0 else ["cuda:0"]
    except Exception:
        return ["cuda:0"]


def dgx_mode_input():
    return ("BOOLEAN", {"default": True, "tooltip": DGX_MODE_TOOLTIP})


def require_cuda_for_dgx_mode(node_name):
    if torch.cuda.is_available():
        return

    raise RuntimeError(
        f"[DGX] {node_name} requires CUDA when DGX mode is enabled. "
        "Turn off DGX mode to use the stock ComfyUI loading pipeline."
    )


def ensure_safetensors_file(path, node_name, fallback_hint):
    if not path.endswith(".safetensors"):
        raise ValueError(
            f"{node_name} only supports .safetensors files. "
            f"Got: {path}. {fallback_hint}"
        )


def load_safetensors_state_dict(path, target_device, load_threads=1):
    load_threads = max(1, int(load_threads))

    if load_threads == 1:
        state_dict = {}
        with safetensors.safe_open(path, framework="pt", device=str(target_device)) as handle:
            metadata = handle.metadata() or {}
            for key in handle.keys():
                state_dict[key] = handle.get_tensor(key)
        return state_dict, metadata

    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {}

    key_groups = [keys[index::load_threads] for index in range(load_threads)]
    state_dict = {}

    def _load_group(key_group):
        local_state_dict = {}
        if not key_group:
            return local_state_dict
        with safetensors.safe_open(path, framework="pt", device=str(target_device)) as handle:
            for key in key_group:
                local_state_dict[key] = handle.get_tensor(key)
        return local_state_dict

    with ThreadPoolExecutor(max_workers=load_threads) as executor:
        for partial_state_dict in executor.map(_load_group, key_groups):
            state_dict.update(partial_state_dict)

    return state_dict, metadata


class AssignOnlyModelPatcher(comfy.model_patcher.ModelPatcher):
    # This keeps ComfyUI's regular ModelPatcher behavior but makes constructor
    # loads use assign=True so CUDA-loaded tensors become the module weights.
    def is_dynamic(self):
        return True


@contextlib.contextmanager
def force_assign_core_model_patcher():
    original = comfy.model_patcher.CoreModelPatcher
    if original is not comfy.model_patcher.ModelPatcher:
        yield
        return

    comfy.model_patcher.CoreModelPatcher = AssignOnlyModelPatcher
    try:
        yield
    finally:
        comfy.model_patcher.CoreModelPatcher = original


@contextlib.contextmanager
def force_text_encoder_devices(target_device):
    original_device = comfy.model_management.text_encoder_device
    original_offload = comfy.model_management.text_encoder_offload_device
    original_dtype = comfy.model_management.text_encoder_dtype
    target_dtype = original_dtype(target_device)

    comfy.model_management.text_encoder_device = lambda *args, **kwargs: target_device
    comfy.model_management.text_encoder_offload_device = lambda *args, **kwargs: target_device
    comfy.model_management.text_encoder_dtype = lambda *args, **kwargs: target_dtype
    try:
        yield
    finally:
        comfy.model_management.text_encoder_device = original_device
        comfy.model_management.text_encoder_offload_device = original_offload
        comfy.model_management.text_encoder_dtype = original_dtype


def gpu_text_encoder_model_options(target_device):
    return {
        "load_device": target_device,
        "offload_device": target_device,
        "initial_device": target_device,
    }


def normalize_clip_metadata_tensors(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if (
            torch.is_tensor(value)
            and value.device.type != "cpu"
            and key.endswith(_CPU_ONLY_CLIP_SUFFIXES)
        ):
            normalized[key] = value.cpu()
        else:
            normalized[key] = value
    return normalized
