"""
SAM3 Segment Tool - ComfyUI Compatibility Shim

Provides drop-in replacements for ComfyUI internal APIs used by SAM3 core code.
This allows the standalone app to run without the full ComfyUI runtime.

Shimmed modules:
  - comfy.ops: manual_cast operations, cast_to_input
  - comfy.model_management: device and memory management
  - comfy.ldm.modules.attention: attention dispatch functions
"""

import logging
import sys
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("sam3_app.shim")


# ═══════════════════════════════════════════════════════════════════════════════
# comfy.ops shim
# ═══════════════════════════════════════════════════════════════════════════════

def cast_to_input(weight, input_tensor):
    """Cast weight to the same dtype and device as the input tensor.
    
    This is a simplified version of comfy.ops.cast_to_input that just
    casts to match the input tensor's dtype and device.
    
    Handles meta-device weights by materializing them first.
    """
    if weight.device.type == "meta":
        # Meta tensor — materialize on the input's device with correct dtype.
        # This can happen if load_state_dict(assign=True) missed some params.
        log.warning(
            "cast_to_input: weight is on meta device, materializing on %s/%s",
            input_tensor.device, input_tensor.dtype,
        )
        weight = torch.empty_like(weight, device=input_tensor.device, dtype=input_tensor.dtype)
        torch.nn.init.xavier_uniform_(weight)
        return weight
    return weight.to(dtype=input_tensor.dtype, device=input_tensor.device)


class ManualCastLinear(nn.Linear):
    """Linear layer that casts weights to match input dtype on forward pass.
    
    Replaces comfy.ops.manual_cast.Linear — ensures weight dtype matches
    activation dtype for mixed-precision inference.
    """
    def forward(self, x):
        weight = cast_to_input(self.weight, x)
        bias = cast_to_input(self.bias, x) if self.bias is not None else None
        return F.linear(x, weight, bias)


class ManualCastConv2d(nn.Conv2d):
    """Conv2d that casts weights to match input dtype on forward pass."""
    def forward(self, x):
        weight = cast_to_input(self.weight, x)
        bias = cast_to_input(self.bias, x) if self.bias is not None else None
        return self._conv_forward(x, weight, bias)


class ManualCastLayerNorm(nn.LayerNorm):
    """LayerNorm that casts weights to match input dtype on forward pass."""
    def forward(self, x):
        weight = cast_to_input(self.weight, x)
        bias = cast_to_input(self.bias, x) if self.bias is not None else None
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class ManualCastEmbedding(nn.Embedding):
    """Embedding that casts weights to match input dtype on forward pass.
    
    Replaces comfy.ops.manual_cast.Embedding — ensures weight dtype matches
    activation dtype for mixed-precision inference.
    """
    def forward(self, x):
        # For meta-device construction, weight may be on meta device
        # During actual inference, we cast to the expected compute dtype
        if self.weight.device.type == "meta":
            # Still on meta device (shouldn't happen after load_state_dict assign=True)
            return F.embedding(x, self.weight, self.padding_idx, self.max_norm,
                               self.norm_type, self.scale_grad_by_freq, self.sparse)
        return F.embedding(x, self.weight.to(self.weight.dtype), self.padding_idx,
                           self.max_norm, self.norm_type, self.scale_grad_by_freq,
                           self.sparse)


class ManualCastConvTranspose2d(nn.ConvTranspose2d):
    """ConvTranspose2d that casts weights to match input dtype on forward pass.
    
    NOTE: Unlike Conv2d, ConvTranspose2d._conv_forward is a type stub that
    returns Ellipsis (...). The real computation happens in forward() via
    F.conv_transpose2d. So we must override forward() directly.
    """
    def forward(self, x, output_size=None):
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose2d")
        
        weight = cast_to_input(self.weight, x)
        bias = cast_to_input(self.bias, x) if self.bias is not None else None
        
        assert isinstance(self.padding, tuple)
        num_spatial_dims = 2
        output_padding = self._output_padding(
            x, output_size,
            self.stride, self.padding, self.kernel_size,
            num_spatial_dims, self.dilation,
        )
        return F.conv_transpose2d(
            x, weight, bias,
            self.stride, self.padding,
            output_padding, self.groups, self.dilation,
        )


class ManualCastGroupNorm(nn.GroupNorm):
    """GroupNorm that casts weights to match input dtype on forward pass."""
    def forward(self, x):
        weight = cast_to_input(self.weight, x)
        bias = cast_to_input(self.bias, x) if self.bias is not None else None
        return F.group_norm(x, self.num_groups, weight, bias, self.eps)


class _ManualCastOps:
    """Module-like object that mimics comfy.ops.manual_cast.
    
    Provides Linear, Conv2d, LayerNorm, Embedding, ConvTranspose2d, GroupNorm
    classes that auto-cast weights for mixed-precision inference.
    """
    Linear = ManualCastLinear
    Conv2d = ManualCastConv2d
    LayerNorm = ManualCastLayerNorm
    Embedding = ManualCastEmbedding
    ConvTranspose2d = ManualCastConvTranspose2d
    GroupNorm = ManualCastGroupNorm

    @staticmethod
    def cast_to_input(weight, input_tensor):
        return cast_to_input(weight, input_tensor)


# ═══════════════════════════════════════════════════════════════════════════════
# comfy.model_management shim
# ═══════════════════════════════════════════════════════════════════════════════

def get_torch_device() -> torch.device:
    """Get the best available torch device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def unet_offload_device() -> torch.device:
    """Device for offloading (CPU)."""
    return torch.device("cpu")


def soft_empty_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# comfy.ldm.modules.attention shim
# ═══════════════════════════════════════════════════════════════════════════════

def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None,
                      skip_reshape=False, skip_output_reshape=False, **kwargs):
    """Standard PyTorch scaled dot-product attention.

    Matches ComfyUI's attention_pytorch signature including skip_reshape
    and skip_output_reshape flags used by SAM3.

    Input / output shapes:
        skip_reshape=False, skip_output_reshape=False:
            q,k,v: [B, N, H*D]  ->  out: [B, N, H*D]
        skip_reshape=True, skip_output_reshape=True:
            q,k,v: [B, H, L, D]  ->  out: [B, H, L, D]
    """
    if skip_reshape:
        # Input already in [B, H, L, D] — just flatten to [B*H, L, D]
        # NOTE: q, k, v may have different sequence lengths (cross-attention),
        # so we must reshape each independently.
        b, h, _, _ = q.shape
        q = q.reshape(b * h, q.shape[2], q.shape[3])
        k = k.reshape(b * h, k.shape[2], k.shape[3])
        v = v.reshape(b * h, v.shape[2], v.shape[3])
        l_q = q.shape[1]
    else:
        # Input in [B, N, H*D] — split heads
        b, n, _ = q.shape
        dim_head = q.shape[-1] // heads
        q = q.reshape(b, n, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, n, dim_head)
        k = k.reshape(b, n, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, n, dim_head)
        v = v.reshape(b, n, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, n, dim_head)
        l_q = n

    dim_head = q.shape[-1]
    scale = dim_head ** -0.5

    # Prepare attention mask for SDPA
    # SDPA expects: None, 2D [L, L], 3D [B*H, L, L], or 4D [B*H, num_heads, L, L]
    # but NOT [1, B*H, L, L] with leading dim 1.
    sdpa_mask = mask
    if sdpa_mask is not None:
        if sdpa_mask.dim() == 4:
            # If mask is [1, B*H, L, L], squeeze the leading dim to [B*H, L, L]
            if sdpa_mask.shape[0] == 1:
                sdpa_mask = sdpa_mask.squeeze(0)
            # If mask is [B*H, 1, L, L] (num_heads=1 broadcast), also squeeze
            elif sdpa_mask.shape[1] == 1 and sdpa_mask.shape[0] == b * h:
                sdpa_mask = sdpa_mask.squeeze(1)
        elif sdpa_mask.dim() == 3 and sdpa_mask.shape[0] == 1:
            # [1, L, L] -> [L, L]
            sdpa_mask = sdpa_mask.squeeze(0)

    if hasattr(F, 'scaled_dot_product_attention'):
        # PyTorch 2.0+ with Flash Attention support
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, scale=scale)
    else:
        # Fallback
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if sdpa_mask is not None:
            attn = attn + sdpa_mask
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)

    # Reshape output
    if skip_output_reshape:
        # Keep heads separate: [B*H, Lq, D] -> [B, H, Lq, D]
        out = out.reshape(b, h, l_q, dim_head)
    else:
        # Merge heads: [B*H, Lq, D] -> [B, Lq, H*D]
        out = out.reshape(b, h, l_q, dim_head).permute(0, 2, 1, 3).reshape(b, l_q, heads * dim_head)

    return out


def optimized_attention_for_device(device, mask=False, small_input=False):
    """Return an attention function suitable for the given device.

    This is a factory function — ComfyUI's real version dispatches to
    Flash Attention, SageAttention, etc. based on device capability.
    Our shim always returns the PyTorch SDPA-based implementation.

    Args:
        device: torch.device (used for dispatch in real ComfyUI)
        mask: If True, return a function that accepts a mask argument
        small_input: If True, prefer simpler attention

    Returns:
        A callable attention function with signature
        fn(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False)
    """
    def _attention_fn(q, k, v, heads, mask=None, attn_precision=None,
                      skip_reshape=False, skip_output_reshape=False, **kwargs):
        return attention_pytorch(q, k, v, heads, mask=mask,
                                 skip_reshape=skip_reshape,
                                 skip_output_reshape=skip_output_reshape)

    # Give the function a recognizable name (SAM3 checks fn.__name__)
    _attention_fn.__name__ = "attention_pytorch"
    return _attention_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Module installation
# ═══════════════════════════════════════════════════════════════════════════════

def install_shims():
    """
    Install ComfyUI shim modules into sys.modules so that
    `import comfy.ops`, `import comfy.model_management`, etc. work
    without the actual ComfyUI runtime.
    
    Safe to call multiple times — no-ops if already installed.
    """
    # ─── comfy ──────────────────────────────────────────────────────────────
    if "comfy" not in sys.modules:
        comfy_mod = types.ModuleType("comfy")
        comfy_mod.__path__ = []  # Make it a package
        comfy_mod.__package__ = "comfy"
        sys.modules["comfy"] = comfy_mod
    else:
        comfy_mod = sys.modules["comfy"]

    # ─── comfy.ops ──────────────────────────────────────────────────────────
    if "comfy.ops" not in sys.modules:
        ops_mod = types.ModuleType("comfy.ops")
        ops_mod.manual_cast = _ManualCastOps()
        ops_mod.cast_to_input = cast_to_input
        ops_mod.__package__ = "comfy.ops"
        sys.modules["comfy.ops"] = ops_mod
        comfy_mod.ops = ops_mod

    # ─── comfy.model_management ─────────────────────────────────────────────
    if "comfy.model_management" not in sys.modules:
        mm_mod = types.ModuleType("comfy.model_management")
        mm_mod.get_torch_device = get_torch_device
        mm_mod.unet_offload_device = unet_offload_device
        mm_mod.soft_empty_cache = soft_empty_cache
        mm_mod.__package__ = "comfy.model_management"
        sys.modules["comfy.model_management"] = mm_mod
        comfy_mod.model_management = mm_mod

    # ─── comfy.utils ────────────────────────────────────────────────────────
    if "comfy.utils" not in sys.modules:
        utils_mod = types.ModuleType("comfy.utils")

        class ProgressBar:
            """Stub for comfy.utils.ProgressBar — no-op in standalone mode."""
            def __init__(self, total=None):
                pass
            def update_absolute(self, value=None):
                pass
            def update(self, value=None):
                pass

        def load_torch_file(ckpt_path):
            """Load a PyTorch or safetensors checkpoint file.
            
            Replaces comfy.utils.load_torch_file — supports both .pt/.pth
            and .safetensors formats.
            """
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(ckpt_path)
            else:
                return torch.load(ckpt_path, map_location="cpu", weights_only=True)

        utils_mod.ProgressBar = ProgressBar
        utils_mod.load_torch_file = load_torch_file
        utils_mod.__package__ = "comfy.utils"
        sys.modules["comfy.utils"] = utils_mod
        comfy_mod.utils = utils_mod

    # ─── comfy.ldm ──────────────────────────────────────────────────────────
    if "comfy.ldm" not in sys.modules:
        ldm_mod = types.ModuleType("comfy.ldm")
        ldm_mod.__path__ = []
        ldm_mod.__package__ = "comfy.ldm"
        sys.modules["comfy.ldm"] = ldm_mod
    else:
        ldm_mod = sys.modules["comfy.ldm"]

    # ─── comfy.ldm.modules ──────────────────────────────────────────────────
    if "comfy.ldm.modules" not in sys.modules:
        ldm_modules = types.ModuleType("comfy.ldm.modules")
        ldm_modules.__path__ = []
        ldm_modules.__package__ = "comfy.ldm.modules"
        sys.modules["comfy.ldm.modules"] = ldm_modules
        ldm_mod.modules = ldm_modules
    else:
        ldm_modules = sys.modules["comfy.ldm.modules"]

    # ─── comfy.ldm.modules.attention ────────────────────────────────────────
    if "comfy.ldm.modules.attention" not in sys.modules:
        attn_mod = types.ModuleType("comfy.ldm.modules.attention")
        attn_mod.optimized_attention_for_device = optimized_attention_for_device
        attn_mod.attention_pytorch = attention_pytorch
        attn_mod.__package__ = "comfy.ldm.modules.attention"
        sys.modules["comfy.ldm.modules.attention"] = attn_mod
        ldm_modules.attention = attn_mod

    # ─── comfy.model_patcher ────────────────────────────────────────────────
    if "comfy.model_patcher" not in sys.modules:
        mp_mod = types.ModuleType("comfy.model_patcher")
        
        class ModelPatcher:
            """Stub for ComfyUI's ModelPatcher — not needed in standalone mode."""
            pass
        
        mp_mod.ModelPatcher = ModelPatcher
        mp_mod.__package__ = "comfy.model_patcher"
        sys.modules["comfy.model_patcher"] = mp_mod
        comfy_mod.model_patcher = mp_mod

    log.info("ComfyUI shims installed (ops, model_management, utils, ldm.modules.attention, model_patcher)")
