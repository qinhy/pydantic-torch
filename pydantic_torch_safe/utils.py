import time
from typing import Any, Optional, Tuple
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


def Cls_parse(v: Any, cls_dict: dict[str, type]) -> Any:
    if isinstance(v, tuple(cls_dict.values())):
        return v
    if not isinstance(v, dict):
        raise TypeError("expected a module instance or serialized module dict")
    raw_uuid = v.get("uuid")
    if not isinstance(raw_uuid, str) or ":" not in raw_uuid:
        raise ValueError("serialized module must include uuid like 'ClassName:...'")
    kind = raw_uuid.split(":", 1)[0]
    module_cls = cls_dict.get(kind)
    if module_cls is None:
        raise ValueError(f"unknown module type: {kind}")
    return module_cls(module_cls.Conf.model_validate(v))

def bind_nested_classes(cls):
    for name, obj in vars(cls).items():
        if isinstance(obj, type) and obj.__module__ == cls.__module__:
            obj.__outer_class__ = cls
    return cls
