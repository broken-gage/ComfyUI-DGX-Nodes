"""
ComfyUI-DGX-Nodes

Standalone DGX Spark / GB10 focused loader nodes for ComfyUI.

Nodes included:
- CheckpointLoaderUnifiedMemory
- UNETLoaderDGX
- CLIPLoaderDGX
- DualCLIPLoaderDGX
- CLIPVisionLoaderDGX
- VAELoaderDGX
- UpscaleModelLoaderDGX
"""

from .checkpoint_loader_unified_memory import CheckpointLoaderUnifiedMemory
from .clip_vision_loader_dgx import CLIPVisionLoaderDGX
from .clip_loader_dgx import CLIPLoaderDGX
from .dual_clip_loader_dgx import DualCLIPLoaderDGX
from .diffusion_model_loader_dgx import UNETLoaderDGX
from .upscale_model_loader_dgx import UpscaleModelLoaderDGX
from .vae_loader_dgx import VAELoaderDGX

NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderUnifiedMemory": CheckpointLoaderUnifiedMemory,
    "UNETLoaderDGX": UNETLoaderDGX,
    "CLIPLoaderDGX": CLIPLoaderDGX,
    "DualCLIPLoaderDGX": DualCLIPLoaderDGX,
    "CLIPVisionLoaderDGX": CLIPVisionLoaderDGX,
    "VAELoaderDGX": VAELoaderDGX,
    "UpscaleModelLoaderDGX": UpscaleModelLoaderDGX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoaderUnifiedMemory": "Checkpoint Loader (Unified Memory)",
    "UNETLoaderDGX": "UNET Loader (Unified Memory)",
    "CLIPLoaderDGX": "CLIP Loader (Unified Memory)",
    "DualCLIPLoaderDGX": "Dual CLIP Loader (Unified Memory)",
    "CLIPVisionLoaderDGX": "CLIP Vision Loader (Unified Memory)",
    "VAELoaderDGX": "VAE Loader (Unified Memory)",
    "UpscaleModelLoaderDGX": "Upscale Model Loader (Unified Memory)",
}

__version__ = "1.2.0"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
