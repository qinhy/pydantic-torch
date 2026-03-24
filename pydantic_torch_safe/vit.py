from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

import torch
from pydantic import Field, field_validator

from pydantic_torch.utils import scaled_dot_product_attention
from pydantic_torch_safe import nn

def _trunc_normal_(t: torch.Tensor, mean: float = 0.0, std: float = 0.02, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    # Prefer native trunc_normal_ if available.
    if hasattr(torch.nn.init, "trunc_normal_"):
        torch.nn.init.trunc_normal_(t, mean=mean, std=std, a=a, b=b)
        return t
    # Fallback: normal then clamp (approx).
    with torch.no_grad():
        t.normal_(mean, std)
        t.clamp_(min=a * std + mean, max=b * std + mean)
    return t


class PatchEmbedNoConv(nn.Module):
    class Conf(nn.Module.Conf):
        img_size: int = Field(default=224, ge=1)
        patch_size: int = Field(default=16, ge=1)
        in_chans: int = Field(default=3, ge=1)
        embed_dim: int = Field(default=768, ge=1)
        num_patches: int = Field(default=0, ge=0)

    def __init__(self, config: Conf | dict):
        super().__init__(config)
        if config.img_size % config.patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        grid = config.img_size // config.patch_size
        config.num_patches = grid * grid

        patch_dim = config.in_chans * config.patch_size * config.patch_size
        self.proj = nn.Linear(in_features=patch_dim, out_features=config.embed_dim)
        self._config:PatchEmbedNoConv.Conf = config
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, patch_dim) -> (B, N, D)
        B, C, H, W = x.shape
        if C != self._config.in_chans:
            raise ValueError(f"Expected in_chans={self._config.in_chans}, got C={C}")
        if H != self._config.img_size or W != self._config.img_size:
            raise ValueError(f"Expected image size {(self._config.img_size, self._config.img_size)}, got {(H, W)}")
        if H % self._config.patch_size != 0 or W % self._config.patch_size != 0:
            raise ValueError("H and W must be divisible by patch_size")

        P = self._config.patch_size
        gh, gw = H // P, W // P

        # Patchify with reshape/permute (no convs).
        x = x.reshape(B, C, gh, P, gw, P)                 # (B, C, gh, P, gw, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()      # (B, gh, gw, C, P, P)
        x = x.reshape(B, gh * gw, C * P * P)              # (B, N, patch_dim)

        x = self.proj(x)                                  # (B, N, D)
        return x


class MLP(nn.Module):
    class Conf(nn.Module.Conf):
        dim: int = Field(default=768, ge=1)
        hidden_dim: int = Field(default=3072, ge=1)
        drop: float = Field(default=0.0, ge=0.0, le=1.0)
        act: nn.Acts.types = Field(default=nn.GELU.Conf())

    def __init__(self, config: Conf | dict = None):
        super().__init__(config)
        config = self.Conf() if config is None else self.Conf.model_validate(config)

        self.fc1 = nn.Linear(nn.Linear.Conf(
            in_features=config.dim,
            out_features=config.hidden_dim,
        ))
        self.fc2 = nn.Linear(nn.Linear.Conf(
            in_features=config.hidden_dim,
            out_features=config.dim,
        ))
        self.dropout = nn.Dropout(nn.Dropout.Conf(p=config.drop))
        self.act = nn.Acts.parse(config.act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    class Conf(nn.Module.Conf):
        dim: int = Field(default=768, ge=1)
        num_heads: int = Field(default=12, ge=1)
        qkv_bias: bool = Field(default=True)
        attn_drop: float = Field(default=0.0, ge=0.0, le=1.0)
        proj_drop: float = Field(default=0.0, ge=0.0, le=1.0)

    def __init__(self, config: Conf | dict = None):
        super().__init__(config)
        config = self.Conf() if config is None else self.Conf.model_validate(config)

        if config.dim % config.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.head_dim = config.dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear({
            "in_features": config.dim,
            "out_features": 3 * config.dim,
            "bias": config.qkv_bias,
        })
        self.proj = nn.Linear({
            "in_features": config.dim,
            "out_features": config.dim,
            "bias": True,
        })
        self.attn_dropout = nn.Dropout({"p": config.attn_drop})
        self.proj_dropout = nn.Dropout({"p": config.proj_drop})

        self._config:Attention.Conf = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape

        out = scaled_dot_product_attention(
            self.qkv(x),
            self._config.num_heads,
            self.head_dim,
            scale=self.scale,
            attn_drop_p=self._config.attn_drop if self.training else 0.0,
        )
        # expected shape: (B, H, N, Hd)

        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out
    
class SelfAttentionBlock(nn.Module):
    class Conf(nn.Module.Conf):
        dim: int = Field(default=768, ge=1)
        num_heads: int = Field(default=12, ge=1)
        mlp_ratio: float = Field(default=4.0, ge=1.0)
        qkv_bias: bool = Field(default=True)
        drop: float = Field(default=0.0, ge=0.0, le=1.0)
        attn_drop: float = Field(default=0.0, ge=0.0, le=1.0)
        drop_path: float = Field(default=0.0, ge=0.0, le=1.0)
        act: nn.Acts.types = Field(default=nn.GELU.Conf())

    def __init__(self, config: Conf | dict = None):
        super().__init__(config)
        config = self.Conf() if config is None else self.Conf.model_validate(config)

        hidden_dim = int(config.dim * config.mlp_ratio)

        self.norm1 = nn.LayerNorm({"normalized_shape": config.dim})
        self.attn = Attention({
            "dim": config.dim,
            "num_heads": config.num_heads,
            "qkv_bias": config.qkv_bias,
            "attn_drop": config.attn_drop,
            "proj_drop": config.drop,
        })
        self.dp1 = nn.DropPath({"drop_prob": config.drop_path})

        self.norm2 = nn.LayerNorm({"normalized_shape": config.dim})
        self.mlp = MLP({
            "dim": config.dim,
            "hidden_dim": hidden_dim,
            "drop": config.drop,
            "act": config.act,
        })
        self.dp2 = nn.DropPath({"drop_prob": config.drop_path})


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x
    
class VisionTransformer(nn.Module):
    class Conf(nn.Module.Conf):
        img_size: int = Field(default=224, ge=1)
        patch_size: int = Field(default=16, ge=1)
        in_chans: int = Field(default=3, ge=1)
        num_classes: int = Field(default=1000, ge=0)

        embed_dim: int = Field(default=768, ge=1)
        depth: int = Field(default=12, ge=1)
        num_heads: int = Field(default=12, ge=1)
        mlp_ratio: float = Field(default=4.0, ge=1.0)

        qkv_bias: bool = Field(default=True)
        drop_rate: float = Field(default=0.0, ge=0.0, le=1.0)
        attn_drop_rate: float = Field(default=0.0, ge=0.0, le=1.0)
        drop_path_rate: float = Field(default=0.0, ge=0.0, le=1.0)
        act: nn.Acts.types = Field(default=nn.GELU.Conf())

    def __init__(self, config: Conf | dict = None):
        super().__init__(config)
        config = self.Conf() if config is None else self.Conf.model_validate(config)

        self.patch_embed = PatchEmbedNoConv({
            "img_size": config.img_size,
            "patch_size": config.patch_size,
            "in_chans": config.in_chans,
            "embed_dim": config.embed_dim,
        })
        num_patches = self.patch_embed._config.num_patches

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, 1 + num_patches, config.embed_dim)
        )
        self.pos_drop = nn.Dropout({"p": config.drop_rate})

        if config.depth == 1:
            dpr = [config.drop_path_rate]
        else:
            dpr = torch.linspace(0.0, config.drop_path_rate, config.depth).tolist()

        self.blocks = torch.nn.ModuleList([
            SelfAttentionBlock({
                "dim": config.embed_dim,
                "num_heads": config.num_heads,
                "mlp_ratio": config.mlp_ratio,
                "qkv_bias": config.qkv_bias,
                "drop": config.drop_rate,
                "attn_drop": config.attn_drop_rate,
                "drop_path": dpr[i],
                "act": config.act,
            })
            for i in range(config.depth)
        ])

        self.norm = nn.LayerNorm({"normalized_shape": config.embed_dim})
        self.head = (
            nn.Linear({
                "in_features": config.embed_dim,
                "out_features": config.num_classes,
            })
            if config.num_classes > 0
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                _trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                if m.weight is not None:
                    torch.nn.init.ones_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, N, D)
        B, N, D = x.shape

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)          # (B, 1+N, D)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        cls = feat[:, 0]
        return self.head(cls)
    
def vit_base_patch16_224(
    num_classes: int = 1000,
    drop_path_rate: float = 0.1,
) -> VisionTransformer:
    return VisionTransformer({
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": drop_path_rate,
    })