import time
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
        qkv: Optional[torch.Tensor] = None,   # (B, N, 3 * H * Hd) , H , Hd
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        query: Optional[torch.Tensor] = None, # (B, H, N, Hd)
        key: Optional[torch.Tensor] = None,   # (B, H, N, Hd)
        value: Optional[torch.Tensor] = None, # (B, H, N, Hd)
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
    ):
    if qkv is not None:
        B, N, _ = qkv.shape
        # qkv = self.qkv(x)  # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]  # (B, H, N, Hd)

    return F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )