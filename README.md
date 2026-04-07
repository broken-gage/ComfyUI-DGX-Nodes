# ComfyUI-DGX-Nodes

Version: 1.0.0


## Author

Repository Owner: broken-gage

Author/Maintainer: Claude Code / CodeX


## Description

Standalone DGX Spark / GB10 focused custom nodes for ComfyUI.

This repo contains experimental unified-memory-aware loader nodes that can load checkpoints, UNETs, CLIP encoders, CLIP vision models, dual-CLIP pairs, and VAEs through a DGX-oriented direct-to-CUDA path. The same nodes also support a stock ComfyUI fallback path through the `dgx_mode` toggle.


## Included Nodes

- `CheckpointLoaderUnifiedMemory`
- `UNETLoaderDGX`
- `CLIPLoaderDGX`
- `DualCLIPLoaderDGX`
- `CLIPVisionLoaderDGX`
- `VAELoaderDGX`


## License

This repository is licensed under the GNU General Public License v3.0. See
[LICENSE](./LICENSE). This matches the GPLv3 licensing used by upstream
ComfyUI.


## Disclaimers

- This is a vibe-code project, and most of the code has been generated or contributed by agentic AI.
- The code is not guaranteed to be complete or fully working in every environment.
- Ongoing maintenance, support, and bug fixes are not guaranteed.
- Use this project at your own risk.


## Tested Environment

- DGX Spark / GB10
- CUDA 13.0
- PyTorch 2.10


## Performance Comparison

Test conditions:
- Clean reboot before switching mode.
- ComfyUI flags: `--gpu-only`, `--cache-none`, `--disable-async-offload`

### Flux.2 Klein 9B Base Image Edit + Turbo 8-step LoRA

| Mode | 1st run | 2nd run | 3rd run | 4th run |
| --- | ---: | ---: | ---: | ---: |
| ComfyUI Native | 172.30s | 55.14s | 55.61s | 56.10s |
| DGX Mode ON | 219.10s | 55.85s | 55.55s | 56.17s |
| DGX Mode OFF | 89.04s | 55.46s | 54.66s | 54.49s |

### Flux.2 Dev FP8 + 8-step Turbo LoRA

| Mode | 1st run | 2nd run | 3rd run | 4th run |
| --- | ---: | ---: | ---: | ---: |
| ComfyUI Native | 462.54s | 62.79s | 24.99s | 29.41s |
| DGX Mode ON | 483.74s | 21.20s | 21.21s | 21.22s |
| DGX Mode OFF | 607.00s | 70.10s | 24.65s | 21.24s |

### Short Conclusion

Results are workload-dependent. In these tests, DGX Mode ON generally had a slower 1st run, but it could improve repeated-run generation speed on some workloads. The benefit is not universal and varies by model and workflow.


## Installation

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

## Platform Notes

- DGX mode is designed for NVIDIA DGX Spark / GB10 systems.
- Fallback mode is intended to remain usable on non-DGX systems, including:
  - Windows x86
  - Ubuntu x86
  - Ubuntu aarch64
- DGX mode requires CUDA. If CUDA is not available, turn `dgx_mode` off.
