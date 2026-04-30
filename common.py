import contextlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import comfy.model_management
import comfy.model_patcher
import safetensors
import torch

try:
    import instanttensor
except Exception:
    instanttensor = None

try:
    from fastsafetensors import fastsafe_open
except Exception:
    fastsafe_open = None

DGX_MODE_TOOLTIP = (
    "ON: use the DGX unified-memory direct-to-CUDA loading path. "
    "OFF: fall back to the stock ComfyUI loading pipeline."
)
CUDA_DEVICE_TOOLTIP = (
    "CUDA device used for the DGX direct-load path when DGX mode is enabled."
)

STORAGE_BACKEND_OPTIONS = ["auto", "instanttensor", "safetensors", "fastsafetensors"]
STORAGE_BACKEND_TOOLTIP = (
    "auto: try instanttensor first (1x memory), then plain safetensors (1x memory), then fastsafetensors (2x peak on unified memory due to host-mmap + CUDA copy). "
    "instanttensor: experimental CUDA safetensors path; load_now=False for minimal peak memory on unified memory. "
    "safetensors: plain safetensors.safe_open(...) path; reads directly into independent CUDA allocations. "
    "fastsafetensors: host-mmap + CUDA DMA path; 2x peak physical memory on unified memory systems."
)

_CPU_ONLY_CLIP_SUFFIXES = (
    "comfy_quant",
    "spiece_model",
    "gemma_spiece_model",
    "jina_spiece_model",
    "tekken_model",
    "mt5xl.spiece_model",
)

_INSTANTTENSOR_GDS_PROBE_RESULT = None


def cuda_device_list():
    try:
        count = torch.cuda.device_count()
        return [f"cuda:{index}" for index in range(count)] if count > 0 else ["cuda:0"]
    except Exception:
        return ["cuda:0"]


def dgx_mode_input():
    return ("BOOLEAN", {"default": True, "tooltip": DGX_MODE_TOOLTIP})


def cuda_device_input():
    return (cuda_device_list(), {"default": "cuda:0", "tooltip": CUDA_DEVICE_TOOLTIP})


def storage_backend_input():
    return (
        STORAGE_BACKEND_OPTIONS,
        {
            "default": "auto",
            "tooltip": STORAGE_BACKEND_TOOLTIP,
        },
    )


def is_safetensors_file(path):
    return path.lower().endswith(".safetensors")


def require_cuda_for_dgx_mode(node_name):
    if torch.cuda.is_available():
        return

    raise RuntimeError(
        f"[DGX] {node_name} requires CUDA when DGX mode is enabled. "
        "Turn off DGX mode to use the stock ComfyUI loading pipeline."
    )


def ensure_safetensors_file(path, node_name, fallback_hint):
    if not is_safetensors_file(path):
        raise ValueError(
            f"{node_name} only supports .safetensors files. "
            f"Got: {path}. {fallback_hint}"
        )


@contextlib.contextmanager
def _temporary_env(var_name, value):
    original = os.environ.get(var_name)
    if value is None:
        os.environ.pop(var_name, None)
    else:
        os.environ[var_name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(var_name, None)
        else:
            os.environ[var_name] = original


def _probe_instanttensor_gds_available():
    global _INSTANTTENSOR_GDS_PROBE_RESULT

    if _INSTANTTENSOR_GDS_PROBE_RESULT is not None:
        return _INSTANTTENSOR_GDS_PROBE_RESULT

    if instanttensor is None or not torch.cuda.is_available():
        _INSTANTTENSOR_GDS_PROBE_RESULT = False
        return _INSTANTTENSOR_GDS_PROBE_RESULT

    probe_script = r"""
import os
import tempfile
import torch
from safetensors.torch import save_file
import instanttensor

fd, path = tempfile.mkstemp(suffix=".safetensors", prefix="instanttensor_probe_")
os.close(fd)
try:
    save_file({"a": torch.arange(12, dtype=torch.float16).reshape(3, 4)}, path)
    with instanttensor.safe_open(
        path,
        framework="pt",
        device="cuda:0",
        load_now=True,
    ) as handle:
        tensor = handle.get_tensor("a")
        assert tensor.device.type == "cuda"
finally:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
"""

    env = os.environ.copy()
    env["INSTANTTENSOR_USE_CUFILE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_script],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        _INSTANTTENSOR_GDS_PROBE_RESULT = result.returncode == 0
    except Exception:
        _INSTANTTENSOR_GDS_PROBE_RESULT = False
    return _INSTANTTENSOR_GDS_PROBE_RESULT


def _load_with_plain_safetensors(path, target_device, load_threads=1):
    load_threads = max(1, int(load_threads))

    if load_threads == 1:
        state_dict = {}
        with safetensors.safe_open(path, framework="pt", device=str(target_device)) as handle:
            metadata = handle.metadata() or {}
            for key in handle.keys():
                state_dict[key] = handle.get_tensor(key)
        return state_dict, metadata, "safetensors", False

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

    return state_dict, metadata, "safetensors", False


def _loader_metadata_or_empty(handle):
    try:
        return handle.metadata() or {}
    except TypeError:
        if hasattr(handle, "file_metadata") and handle.file_metadata is None:
            return {}
        raise


def _load_with_instanttensor(path, target_device):
    if instanttensor is None:
        raise RuntimeError("instanttensor is not installed in the active environment.")

    use_gds = _probe_instanttensor_gds_available()
    state_dict = {}
    # load_now=False: lazy per-tensor loading — each get_tensor() reads directly into
    # an independent CUDA allocation with no pre-loaded internal buffer.
    # This keeps peak physical memory at 1x on unified memory (no buffer+clones overlap).
    with _temporary_env("INSTANTTENSOR_USE_CUFILE", "1" if use_gds else "0"):
        with instanttensor.safe_open(
            path,
            framework="pt",
            device=str(target_device),
            load_now=False,
        ) as handle:
            metadata = _loader_metadata_or_empty(handle)
            for key in handle.keys():
                # clone() is required: instanttensor tensors are backed by
                # context-managed memory freed on exit. With load_now=False the
                # clone covers one tensor at a time (not the whole model), so
                # peak overhead is one tensor rather than 1× model size.
                state_dict[key] = handle.get_tensor(key).clone()
    return state_dict, metadata, "instanttensor", bool(use_gds)


def _extract_fastsafetensors_metadata(metadata_by_file, path):
    if not metadata_by_file:
        return {}
    if isinstance(metadata_by_file, dict):
        if path in metadata_by_file and isinstance(metadata_by_file[path], dict):
            return metadata_by_file[path]
        first_value = next(iter(metadata_by_file.values()), {})
        return first_value if isinstance(first_value, dict) else {}
    return {}


def _load_with_fastsafetensors(path, target_device):
    if fastsafe_open is None:
        raise RuntimeError("fastsafetensors is not installed in the active environment.")

    state_dict = {}
    with fastsafe_open(
        path,
        framework="pt",
        device=str(target_device),
        nogds=True,
        debug_log=False,
    ) as handle:
        metadata = _extract_fastsafetensors_metadata(handle.metadata(), path)
        for key in handle.keys():
            state_dict[key] = handle.get_tensor(key).clone()
        # fastsafe_open pre-allocates the entire file as a single CUDA buffer at init.
        # get_tensor() returns views of that buffer; .clone() makes independent copies.
        # The buffer (N GB) + accumulated clones (N GB) sit in CUDA simultaneously until
        # __exit__ frees the buffer. Closing it here — right after all clones exist —
        # shrinks the 2× window to just the clone loop instead of lingering until exit.
        # fastsafe_open.__exit__ guards with `if self.fb:` so setting fb=None is safe.
        if getattr(handle, "fb", None) is not None:
            handle.fb.close()
            handle.fb = None
    return state_dict, metadata, "fastsafetensors", False


def load_safetensors_state_dict(path, target_device, load_threads=1, storage_backend="auto"):
    if storage_backend not in STORAGE_BACKEND_OPTIONS:
        raise ValueError(
            f"Unsupported storage backend: {storage_backend}. "
            f"Expected one of: {', '.join(STORAGE_BACKEND_OPTIONS)}"
        )

    if storage_backend == "auto":
        # Prefer backends with 1x peak physical memory on unified memory systems.
        # instanttensor (load_now=False): truly lazy, no internal buffer.
        # safetensors: direct CUDA writes, small OS page-cache window only.
        # fastsafetensors: host-mmap + CUDA DMA → 2x physical peak, last resort.
        backend_order = ["instanttensor", "safetensors", "fastsafetensors"]
    else:
        backend_order = [storage_backend]

    errors = {}
    for backend in backend_order:
        try:
            if backend == "instanttensor":
                return _load_with_instanttensor(path, target_device)
            if backend == "fastsafetensors":
                return _load_with_fastsafetensors(path, target_device)
            if backend == "safetensors":
                return _load_with_plain_safetensors(path, target_device, load_threads=load_threads)
        except Exception as exc:
            errors[backend] = repr(exc)
            if storage_backend != "auto":
                raise

    error_summary = "; ".join(f"{name}={message}" for name, message in errors.items())
    raise RuntimeError(
        f"Failed to load safetensors file via the available DGX backends: {error_summary}"
    )


class AssignOnlyModelPatcher(comfy.model_patcher.ModelPatcher):
    # This keeps ComfyUI's regular ModelPatcher behavior but makes constructor
    # loads use assign=True so CUDA-loaded tensors become the module weights.
    def is_dynamic(self):
        return True


@contextlib.contextmanager
def force_assign_core_model_patcher():
    # Always replace CoreModelPatcher with AssignOnlyModelPatcher, even when
    # aimdo (dynamic VRAM) is active and has set CoreModelPatcher = ModelPatcherDynamic.
    #
    # Without this, CLIP and VAE get ModelPatcherDynamic patchers. Aimdo's
    # ModelPatcherDynamic.partially_load() has no skip condition — it always
    # calls load(), which allocates a ~1x model-size CUDA staging buffer and
    # resets model_loaded_weight_memory to only the pre-loaded weights (~600 KB).
    # This causes future model_memory_required() calls to return the full model
    # size, triggering free_memory() that evicts other models (e.g. the UNet).
    #
    # AssignOnlyModelPatcher uses base ModelPatcher.partially_load() which
    # respects mark_patcher_as_loaded()'s pre-set model_loaded_weight_memory,
    # so the skip condition fires and no staging buffer is ever allocated.
    original = comfy.model_patcher.CoreModelPatcher
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
    # initial_device=meta: the text encoder skeleton is created with meta tensors
    # (zero physical memory). assign=True in load_sd() then directly replaces them
    # with the already-CUDA sd tensors — peak stays at 1x on unified memory.
    #
    # Side effect: meta != load_device, so CLIP.__init__ does NOT call
    # load_models_gpu internally (condition: params['device'] == load_device is False).
    # This prevents the spurious free_memory() call that was unnecessarily evicting
    # other models when dynamic VRAM (aimdo) made loaded_size() report 0 before
    # partially_load() had run.
    return {
        "load_device": target_device,
        "offload_device": target_device,
        "initial_device": torch.device("meta"),
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


def mark_patcher_as_loaded(patcher, device):
    """Update model_management tracking state after assign=True weight loading.

    load_model_weights(assign=True) puts weights directly on the target device
    without going through ModelPatcher.load(), so model_loaded_weight_memory stays
    at 0 and model.device stays at offload_device. load_models_gpu() reads these
    fields before calling free_memory(): with stale values it thinks the full model
    still needs to be loaded and unnecessarily evicts other models.

    This corrects the counters so load_models_gpu sees model_memory_required=0 and
    skips free_memory(). The subsequent ModelPatcher.load() / ModelPatcherDynamic.load()
    call (inside load_models_gpu) will overwrite these values with its own tracking.
    """
    patcher.model.model_loaded_weight_memory = patcher.model_size()
    patcher.model.device = device
