# ComfyUI-DGX-Nodes

Version: 1.2.1

Release Date: 2026-05-01


## Author

Repository Owner: broken-gage

Author/Maintainer: Claude Code / CodeX


## Description

Standalone DGX Spark / GB10 focused custom nodes for ComfyUI.

This repo contains experimental unified-memory-aware loader nodes that can load checkpoints, UNETs, CLIP encoders, CLIP vision models, dual-CLIP pairs, VAEs, and upscaler models through a DGX-oriented direct-to-CUDA path. The same nodes also support a stock ComfyUI fallback path through the `dgx_mode` toggle.

Currently, `dgx_mode` only works with NVIDIA / CUDA systems. It should also work with discrete NVIDIA graphics cards, but that has not been validated. Trying to load model weights larger than the available VRAM when using `dgx_mode` may cause system instability and OOM errors. We recommend using the `--disable-mmap` startup flag only with DGX Spark / GB10 / Jetson systems.


## Included Nodes

- `CheckpointLoaderUnifiedMemory`
- `UNETLoaderDGX`
- `CLIPLoaderDGX`
- `DualCLIPLoaderDGX`
- `CLIPVisionLoaderDGX`
- `VAELoaderDGX`
- `UpscaleModelLoaderDGX`


## License

This repository is licensed under the GNU General Public License v3.0. See
[LICENSE](./LICENSE). This matches the GPLv3 licensing used by upstream
ComfyUI.


## Disclaimers

- This is a vibe-code project, and most of the code has been generated or contributed by agentic AI.
- The code is not guaranteed to be complete or fully working in every environment.
- Ongoing maintenance, support, and bug fixes are not guaranteed.
- Use this project at your own risk.

---

## Version `1.2.1` updates:

- Package metadata name reverted to `dgx-gb10-nodes` for Comfy Registry compatibility.
- Removed the install-time legacy folder cleanup script that was only needed for the package name change.

## Version `1.2.0` updates:

- Updated nodes to be compatible with the new ComfyUI dynamic VRAM.
- `instanttensor` is now implemented correctly, with safer `instanttensor` metadata handling for safetensors files without metadata.
- Backend selector ordering aligned with the automatic backend priority.
- Requires ComfyUI version `0.20.1` or newer, which supports dynamic VRAM. For ComfyUI versions before `0.20.0`, use node version `1.1.0`.

## Known Issue
- With ComfyUI's dynamic VRAM implementation, the DGX node backends use more peak VRAM than native loading nodes (up to a 30% transient RAM usage increase in some cases).

## Performance Results - Version 1.2.0

Test conditions:
- Python 3.12.3
- DGX Spark / GB10 \*
- CUDA 13.0
- Clean reboot before switching mode.
- ComfyUI flags: `--disable-mmap`
- Image edit, 1024 x 1024
- UNET, CLIP, and VAE nodes are replaced with DGX Nodes
- Native path tested with official ComfyUI nodes

\* For some reason, the machine was stuck in 30W safe mode. For result consistency, all tests were performed under safe mode.

### Flux.2 Dev FP8 Mixed - Text to Image + small decoder + 8-step Turbo LoRA - Ver 1.2.0

| Mode | 1st run | 2nd run | 3rd run | 4th run | Est. Load Time | Memory Peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ComfyUI Native | 399.17s | 46.85s | 46.27s | 46.45s | 352.65s | 74 GiB |
| fastsafetensors | 90.03s | 47.11s | 46.30s | 46.28s | 43.47s | 107 GiB |
| instanttensor | 73.03s | 46.31s | 46.27s | 46.32s | 26.73s | 78 GiB |
| safetensors\*\* | - s | - s | - s | - s | - s | - GiB |
| DGX mode off | 404.91s | 47.52s | 46.39s | 46.38s | 358.15s | 74 GiB |

### Flux.2 Klein 9B Base FP8 - Text to Image + small decoder + 8-step Turbo LoRA - Ver 1.2.0

| Mode | 1st run | 2nd run | 3rd run | 4th run | Est. Load Time | Memory Peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ComfyUI Native | 91.31s | 28.33s | 28.35s | 28.35s | 62.97s | 27 GiB |
| fastsafetensors | 39.52s | 28.32s | 28.43s | 28.33s | 11.16s | 32 GiB |
| instanttensor | 42.20s | 28.35s | 28.43s | 28.37s | 13.82s | 30 GiB |
| safetensors\*\* | - s | - s | - s | - s | - s | - GiB |
| DGX mode off | 91.78s | 28.31s | 28.29s | 28.26s | 63.49s | 28 GiB |

\*\* The safetensors backend has an implementation issue that causes unintended model weight unloading, so it is excluded from test results.

### Short Conclusion - Version 1.2.0
With ComfyUI fixing the RAM double-loading issue with dynamic VRAM, and with the implementation of the `instanttensor` pipeline, the load time results changed significantly compared to v1.1.0.

With `instanttensor`, the model loading speed is about 4x (~5 GB/s) compared to `fastsafetensors` (~1.2 GB/s). The generation / inference time is similar across all loading methods, which means inference performance is identical across all loading methods.

---

Version `1.1.0` adds:

- `UpscaleModelLoaderDGX`
- backend-aware safetensors loading with `instanttensor`, `fastsafetensors`, and plain `safetensors`
- stock fallback for unsupported upscaler formats such as `.pth`

The current backend order is:

- `instanttensor`
- `fastsafetensors`
- plain `safetensors`

The `instanttensor` pipeline is currently a work in progress and experimental. It is wired into the node package, but it still needs more validation and stabilization on real large-model workloads. It is highly recommended to select `fastsafetensors` when using a DGX Spark / GB10 system.

## Tested Environment

- DGX Spark / GB10
- CUDA 13.0
- PyTorch 2.10


## Performance Comparison

Test conditions:
- Clean reboot before switching mode.
- ComfyUI flags: `--gpu-only`, `--cache-none`, `--disable-async-offload`
- Image edit, 1024 x 1024
- UNET, CLIP, and VAE nodes are replaced with DGX Nodes

### Flux.2 Dev FP8 Mixed - Single Image Edit + 8-step Turbo LoRA - Ver 1.1.0

| Mode | 1st run | 2nd run | 3rd run | 4th run |
| --- | ---: | ---: | ---: | ---: |
| ComfyUI Native\* | 756.47s | 141.57s | 114.50s | 99.44s |
| fastsafetensors | 214.19s | 99.06s | 99.27s | 99.27s |

\* Model loading and inference performance may be affected by RAM double loading, which caused spill to swap and affected test results.

### Flux.2 Klein 9B Base FP8 - Single Image Edit + 8-step Turbo LoRA - Ver 1.1.0

| Mode | 1st run | 2nd run | 3rd run | 4th run |
| --- | ---: | ---: | ---: | ---: |
| ComfyUI Native | 217.91s | 31.86s | 31.88s | 31.88s |
| fastsafetensors | 55.84s | 31.82s | 31.80s | 31.78s |


### Short Conclusion

The `fastsafetensors` pipeline is now able to load models directly to VRAM, bypassing the CPU/RAM staging process. However, the improvement in overall inference time is workload-dependent. While some workloads benefit from shorter model load time, other workloads may experience a slowdown in load time and inference time depending on the nature of the downstream nodes and pipelines.

| Workflow Model | Execution Time Improvement |
| --- |  ---: |
| Flux.2-Dev FP8 Mixed | Positive |
| Flux.2-Klein 9B Base | Positive |
| Qwen-Image-Edit (2511) | Positive |
| WAN 2.2 T2V 14B | Negative |
| WAN 2.2 I2V 14B | Negative |

---

## Installation

We recommend using ComfyUI Manager to install these custom nodes. Alternatively, you can also install them manually with the following steps:

1. Place this repo under `ComfyUI/custom_nodes/ComfyUI-DGX-Nodes`
2. Install dependencies with `pip install -r requirements.txt`
3. Restart ComfyUI
4. The nodes will appear under the `DGX Nodes` category

No ComfyUI core-file modifications are required.

## Usage

- `dgx_mode=ON`
  Uses the DGX direct-to-CUDA unified-memory loading path. This is intended for
  DGX Spark / GB10 systems.

- `dgx_mode=OFF`
  Falls back to the stock ComfyUI loading path. This keeps the same workflow
  node identities usable on non-DGX systems, including x86 platforms.

- `storage_backend=auto`
  (Experimental) Tries the DGX backend stack in order and falls back safely when a higher-performance backend is unavailable or unsuitable.

- `storage_backend=instanttensor`
  (Experimental) Uses the `instanttensor` loader path for safetensors files. This path is currently experimental and should be treated as work in progress.

- `storage_backend=fastsafetensors`
  Uses `fastsafetensors` for safetensors files. On GB10 this is currently integrated in no-GDS mode by default because the library's built-in GDS platform detection does not currently line up with this machine.

- `storage_backend=safetensors`
  Uses the existing plain `safetensors.safe_open(...)` direct-to-CUDA path.
  **Not recommended. In most cases, it is better to use the stock loading path by turning `dgx_mode` off.**

- `Upscale Model Loader (Unified Memory)`
  Mirrors stock `Load Upscale Model`. Safetensors upscaler models use the DGX backend stack when possible; unsupported formats such as `.pth` automatically fall back to the conventional ComfyUI loader.

## Platform Notes

- DGX mode is designed for NVIDIA DGX Spark / GB10 systems.
- GDS-capable user-space/runtime components are present on the target GB10 platform, but individual third-party loader libraries can still need backend-specific handling or safe fallback on this stack.
- The `instanttensor` backend is currently integrated as an experimental path and may still require additional tuning or fallback on large-model workloads.
- Fallback mode is intended to remain usable on non-DGX systems, including:
  - Windows x86
  - Ubuntu x86
  - Ubuntu aarch64
- DGX mode requires CUDA. If CUDA is not available, turn `dgx_mode` off.
