from __future__ import annotations

from typing import Any, Optional

import torch
from pydantic import Field
from . import modules as nn
from .containers import ModuleList

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
    img_size: int = Field(default=224, ge=1)
    patch_size: int = Field(default=16, ge=1)
    in_chans: int = Field(default=3, ge=1)
    embed_dim: int = Field(default=768, ge=1)

    num_patches: int = Field(default=0, ge=0)

    proj: nn.Linear = Field(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.img_size % self.patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        grid = self.img_size // self.patch_size
        self.num_patches = grid * grid

        patch_dim = self.in_chans * self.patch_size * self.patch_size
        self.proj = nn.Linear(in_features=patch_dim, out_features=self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, patch_dim) -> (B, N, D)
        B, C, H, W = x.shape
        if C != self.in_chans:
            raise ValueError(f"Expected in_chans={self.in_chans}, got C={C}")
        if H != self.img_size or W != self.img_size:
            raise ValueError(f"Expected image size {(self.img_size, self.img_size)}, got {(H, W)}")
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError("H and W must be divisible by patch_size")

        P = self.patch_size
        gh, gw = H // P, W // P

        # Patchify with reshape/permute (no convs).
        x = x.reshape(B, C, gh, P, gw, P)                 # (B, C, gh, P, gw, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()      # (B, gh, gw, C, P, P)
        x = x.reshape(B, gh * gw, C * P * P)              # (B, N, patch_dim)

        x = self.proj(x)                                  # (B, N, D)
        return x


class MLP(nn.Module):
    dim: int = Field(default=768, ge=1)
    hidden_dim: int = Field(default=3072, ge=1)
    drop: float = Field(default=0.0, ge=0.0, le=1.0)

    fc1: nn.Linear = Field(default=None)
    fc2: nn.Linear = Field(default=None)
    act: nn.GELU = Field(default=None)
    dropout: nn.Dropout = Field(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.fc1 = nn.Linear(in_features=self.dim, out_features=self.hidden_dim)
        self.fc2 = nn.Linear(in_features=self.hidden_dim, out_features=self.dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=self.drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    dim: int = Field(default=768, ge=1)
    num_heads: int = Field(default=12, ge=1)
    qkv_bias: bool = Field(default=True)
    attn_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    proj_drop: float = Field(default=0.0, ge=0.0, le=1.0)

    qkv: nn.Linear = Field(default=None)
    proj: nn.Linear = Field(default=None)
    attn_dropout: nn.Dropout = Field(default=None)
    proj_dropout: nn.Dropout = Field(default=None)

    head_dim: int = Field(default=0, ge=0)
    scale: float = Field(default=0.0)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(in_features=self.dim, out_features=3 * self.dim, bias=self.qkv_bias)
        self.proj = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.attn_dropout = nn.Dropout(p=self.attn_drop)
        self.proj_dropout = nn.Dropout(p=self.proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x)  # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, H, N, Hd)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class SelfAttentionBlock(nn.Module):
    dim: int = Field(default=768, ge=1)
    num_heads: int = Field(default=12, ge=1)
    mlp_ratio: float = Field(default=4.0, ge=1.0)
    qkv_bias: bool = Field(default=True)
    drop: float = Field(default=0.0, ge=0.0, le=1.0)
    attn_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    drop_path: float = Field(default=0.0, ge=0.0, le=1.0)

    norm1: nn.LayerNorm = Field(default=None)
    norm2: nn.LayerNorm = Field(default=None)
    attn: Attention = Field(default=None)
    mlp: MLP = Field(default=None)
    dp1: nn.DropPath = Field(default=None)
    dp2: nn.DropPath = Field(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.norm1 = nn.LayerNorm(normalized_shape=self.dim)
        self.attn = Attention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
        )
        self.dp1 = nn.DropPath(drop_prob=self.drop_path)

        self.norm2 = nn.LayerNorm(normalized_shape=self.dim)
        hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(dim=self.dim, hidden_dim=hidden_dim, drop=self.drop)
        self.dp2 = nn.DropPath(drop_prob=self.drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
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

    patch_embed: PatchEmbedNoConv = Field(default=None)
    pos_drop: nn.Dropout = Field(default=nn.Dropout)
    blocks: ModuleList = Field(default_factory=ModuleList)
    norm: nn.LayerNorm = Field(default=None)
    head: nn.Linear = Field(default=None)

    # Parameters (learned)
    cls_token: Optional[torch.nn.Parameter] = Field(default=None, exclude=True)
    pos_embed: Optional[torch.nn.Parameter] = Field(default=None, exclude=True)

    num_patches: int = Field(default=0, ge=0)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        self.patch_embed = PatchEmbedNoConv(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 1 + self.num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # linearly increasing stochastic depth
        if self.depth == 1:
            dpr = [self.drop_path_rate]
        else:
            dpr = torch.linspace(0.0, self.drop_path_rate, self.depth).tolist()

        self.blocks = ModuleList(mods=[
            SelfAttentionBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(normalized_shape=self.embed_dim)
        self.head = nn.Linear(in_features=self.embed_dim, out_features=self.num_classes) if self.num_classes > 0 else nn.Identity()

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
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)               # (B, N, D)
        B, N, D = x.shape

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)          # (B, 1+N, D)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.head(feat)


def vit_base_patch16_224(num_classes: int = 1000, drop_path_rate: float = 0.1) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
    )


