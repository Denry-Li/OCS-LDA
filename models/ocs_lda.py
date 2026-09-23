"""Final OCS-LDA model, assembled from the archived V3 and V4 sources.

The historical class name PhySP_AEv4 is retained for checkpoint compatibility.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from models.ocs_lda_blocks import (
    BranchLatentModule,
    GroupStemBank,
    RSTB,
    StructuredSwinAE,
    SwinFeatureStage,
    ceil_to_multiple,
    cfg_get,
)

Tensor = torch.Tensor


def _as_int_list(value, name: str) -> list[int]:
    if isinstance(value, (list, tuple)):
        result = [int(v) for v in value]
    else:
        result = [int(value)]
    if not result or any(v <= 0 for v in result):
        raise ValueError(f"{name} must contain positive integers.")
    return result


def _as_nonnegative_int_list(value, name: str) -> list[int]:
    if isinstance(value, (list, tuple)):
        result = [int(v) for v in value]
    else:
        result = [int(value)]
    if not result or any(v < 0 for v in result):
        raise ValueError(f"{name} must contain non-negative integers.")
    return result


def _resolve_group_config(value, group_name: str, default):
    if value is None:
        return default
    if isinstance(value, Mapping):
        if group_name in value:
            return value[group_name]
        if "default" in value:
            return value["default"]
        return default
    return value


def _compatible_num_heads(dim: int, preferred: int) -> int:
    for heads in range(min(int(dim), int(preferred)), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _lcm(*values: int) -> int:
    result = 1
    for value in values:
        result = math.lcm(result, int(value))
    return result


def _input_alignment(params) -> int:
    stem_downsample = int(cfg_get(params, "stem_downsample", 1))
    encoder_merges = int(cfg_get(params, "encoder_merges", 0))
    window_size = int(cfg_get(params, "window_size", 8))
    latent_downsample = int(cfg_get(params, "latent_downsample", 4))
    bottleneck_window = int(
        cfg_get(params, "bottleneck_window_size", min(4, window_size))
    )
    decoder_window = int(
        cfg_get(
            params,
            "decoder_window_size",
            cfg_get(params, "group_refine_window_size", window_size),
        )
    )
    encoder_factor = stem_downsample * (2**encoder_merges)
    return _lcm(
        encoder_factor * window_size,
        latent_downsample * bottleneck_window,
        latent_downsample * decoder_window,
    )


def _pad_geophysical_field(
    x: Tensor,
    target_h: int,
    target_w: int,
    longitude_periodic: bool,
    latitude_mode: str,
) -> Tensor:
    """Pad on the high-index sides while respecting lon/lat boundary semantics."""
    h, w = x.shape[-2:]
    pad_h, pad_w = int(target_h) - h, int(target_w) - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"Target size {(target_h, target_w)} is smaller than input {(h, w)}."
        )
    if latitude_mode not in {"reflect", "replicate"}:
        raise ValueError("encoder_latitude_padding must be 'reflect' or 'replicate'.")
    if pad_w:
        width_mode = "circular" if longitude_periodic else "replicate"
        x = F.pad(x, (0, pad_w, 0, 0), mode=width_mode)
    if pad_h:
        mode = latitude_mode
        if mode == "reflect" and pad_h >= x.shape[-2]:
            mode = "replicate"
        x = F.pad(x, (0, 0, 0, pad_h), mode=mode)
    return x


class ProgressiveStrideStem(nn.Module):
    """Channel projection with optional progressive 2x/4x spatial reduction."""

    def __init__(self, in_ch: int, out_ch: int, downsample: int):
        super().__init__()
        self.downsample = int(downsample)
        if self.downsample not in {1, 2, 4}:
            raise ValueError("stem_downsample must be one of 1, 2, 4.")
        if self.downsample == 1:
            self.net = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            return

        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        ]
        if self.downsample == 4:
            layers.extend(
                [
                    nn.Conv2d(
                        out_ch, out_ch, kernel_size=3, stride=2, padding=1
                    ),
                    nn.GELU(),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class InterStagePatchMerging(nn.Module):
    """Merge 2x2 tokens between stages with fixed or doubled output channels."""

    def __init__(self, dim: int, dim_mode: str, norm_layer=nn.LayerNorm):
        super().__init__()
        if dim_mode not in {"fixed", "double"}:
            raise ValueError("encoder_dim_mode must be 'fixed' or 'double'.")
        self.dim = int(dim)
        self.out_dim = self.dim if dim_mode == "fixed" else self.dim * 2
        self.norm = norm_layer(4 * self.dim)
        self.reduction = nn.Linear(4 * self.dim, self.out_dim, bias=False)

    def forward(
        self, x: Tensor, spatial_size: Tuple[int, int]
    ) -> Tuple[Tensor, Tuple[int, int]]:
        h, w = spatial_size
        b, length, channels = x.shape
        if length != h * w or channels != self.dim:
            raise RuntimeError(
                "PatchMerging state mismatch: "
                f"x={tuple(x.shape)}, spatial_size={spatial_size}, dim={self.dim}."
            )
        if h % 2 or w % 2:
            raise RuntimeError(
                f"PatchMerging requires even H/W, got spatial_size={spatial_size}."
            )

        x = x.view(b, h, w, channels)
        x = torch.cat(
            (
                x[:, 0::2, 0::2, :],
                x[:, 1::2, 0::2, :],
                x[:, 0::2, 1::2, :],
                x[:, 1::2, 1::2, :],
            ),
            dim=-1,
        )
        x = x.reshape(b, -1, 4 * channels)
        x = self.reduction(self.norm(x))
        return x, (h // 2, w // 2)


class HierarchicalSwinFeatureEncoder(nn.Module):
    """Same-resolution RSTB stages separated by independent PatchMerging."""

    def __init__(
        self,
        params,
        input_size: Tuple[int, int],
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        if int(cfg_get(params, "patch_size", 1)) != 1:
            raise ValueError("PhySP_AEv3 requires patch_size=1.")

        self.embed_dim = int(cfg_get(params, "embed_dim", 64))
        self.encoder_merges = int(cfg_get(params, "encoder_merges", 0))
        self.dim_mode = str(cfg_get(params, "encoder_dim_mode", "fixed"))
        depths = _as_int_list(
            cfg_get(params, "encoder_depths", [2]), "encoder_depths"
        )
        heads = _as_int_list(
            cfg_get(params, "encoder_num_heads", [4]), "encoder_num_heads"
        )
        if self.encoder_merges < 0:
            raise ValueError("encoder_merges must be >= 0.")
        if len(depths) != self.encoder_merges + 1:
            raise ValueError(
                "len(encoder_depths) must equal encoder_merges + 1; "
                f"got depths={depths}, encoder_merges={self.encoder_merges}."
            )
        if len(heads) != len(depths):
            raise ValueError(
                "encoder_num_heads and encoder_depths must have equal length."
            )
        if self.dim_mode not in {"fixed", "double"}:
            raise ValueError("encoder_dim_mode must be 'fixed' or 'double'.")

        self.stage_dims = [
            self.embed_dim * (2**i if self.dim_mode == "double" else 1)
            for i in range(len(depths))
        ]
        self.stage_resolutions = [
            (input_size[0] // (2**i), input_size[1] // (2**i))
            for i in range(len(depths))
        ]
        for i, (dim, num_heads) in enumerate(zip(self.stage_dims, heads)):
            if dim % num_heads:
                raise ValueError(
                    f"Stage {i}: dim={dim} must be divisible by num_heads={num_heads}."
                )

        drop_path_rate = float(cfg_get(params, "drop_path_rate", 0.0))
        stage_rates = torch.linspace(0, drop_path_rate, len(depths)).tolist()
        self.token_norm = (
            norm_layer(self.embed_dim)
            if bool(cfg_get(params, "patch_norm", True))
            else nn.Identity()
        )
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        for i, depth in enumerate(depths):
            dim = self.stage_dims[i]
            resolution = self.stage_resolutions[i]
            self.stages.append(
                RSTB(
                    dim=dim,
                    input_resolution=resolution,
                    depth=depth,
                    num_heads=heads[i],
                    window_size=int(cfg_get(params, "window_size", 8)),
                    mlp_ratio=float(cfg_get(params, "mlp_ratio", 2.0)),
                    qkv_bias=bool(cfg_get(params, "qkv_bias", True)),
                    qk_scale=cfg_get(params, "qk_scale", None),
                    drop=float(cfg_get(params, "drop_rate", 0.0)),
                    attn_drop=float(cfg_get(params, "attn_drop_rate", 0.0)),
                    drop_path=[stage_rates[i]] * depth,
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=bool(cfg_get(params, "use_checkpoint", False)),
                    img_size=resolution,
                    patch_size=1,
                    resi_connection=str(
                        cfg_get(params, "resi_connection", "1conv")
                    ),
                )
            )
            if i < self.encoder_merges:
                self.merges.append(
                    InterStagePatchMerging(
                        dim=dim,
                        dim_mode=self.dim_mode,
                        norm_layer=norm_layer,
                    )
                )
        self.norm = norm_layer(self.stage_dims[-1])
        self.out_channels = self.stage_dims[-1]

    def forward(self, x: Tensor) -> Tensor:
        b, channels, h, w = x.shape
        if channels != self.embed_dim:
            raise RuntimeError(
                f"Expected encoder channels={self.embed_dim}, got {channels}."
            )
        spatial_size = (h, w)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.token_norm(x)

        for i, stage in enumerate(self.stages):
            expected_dim = self.stage_dims[i]
            if x.shape[1] != spatial_size[0] * spatial_size[1]:
                raise RuntimeError(
                    f"Stage {i}: token count does not match {spatial_size}."
                )
            if x.shape[2] != expected_dim:
                raise RuntimeError(
                    f"Stage {i}: expected channels={expected_dim}, got {x.shape[2]}."
                )
            x = stage(x, spatial_size)
            if i < len(self.merges):
                x, spatial_size = self.merges[i](x, spatial_size)

        x = self.norm(x)
        h, w = spatial_size
        return x.transpose(1, 2).contiguous().view(b, self.out_channels, h, w)


class SharedHierarchicalSwinEncoder(nn.Module):
    """Grouped-to-shared projection followed by the hierarchical Swin encoder."""

    def __init__(self, params, in_ch: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dim = int(cfg_get(params, "embed_dim", 64))
        self.stem_downsample = int(cfg_get(params, "stem_downsample", 1))
        self.input_size = (
            int(cfg_get(params, "img_size_x")),
            int(cfg_get(params, "img_size_y")),
        )
        alignment = _input_alignment(params)
        padded_size = (
            ceil_to_multiple(self.input_size[0], alignment),
            ceil_to_multiple(self.input_size[1], alignment),
        )
        stem_size = (
            padded_size[0] // self.stem_downsample,
            padded_size[1] // self.stem_downsample,
        )
        self.input_projection = ProgressiveStrideStem(
            in_ch=in_ch,
            out_ch=self.embed_dim,
            downsample=self.stem_downsample,
        )
        self.backbone = HierarchicalSwinFeatureEncoder(
            params=params,
            input_size=stem_size,
            norm_layer=norm_layer,
        )
        self.out_channels = self.backbone.out_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(self.input_projection(x))


class HierarchicalStructuredBottleneck(nn.Module):
    """Complete the requested downsampling, project to latent, and split branches."""

    def __init__(
        self,
        params,
        encoder_out_channels: int,
        branch_channels: Mapping[str, int],
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.latent_dim = int(cfg_get(params, "latent_dim", 64))
        self.latent_downsample = int(cfg_get(params, "latent_downsample", 4))
        self.stem_downsample = int(cfg_get(params, "stem_downsample", 1))
        self.encoder_merges = int(cfg_get(params, "encoder_merges", 0))
        if not _is_power_of_two(self.latent_downsample):
            raise ValueError("latent_downsample must be a positive power of two.")

        encoder_factor = self.stem_downsample * (2**self.encoder_merges)
        if self.latent_downsample % encoder_factor:
            raise ValueError(
                f"latent_downsample={self.latent_downsample} must be divisible by "
                f"encoder factor={encoder_factor}."
            )
        remaining = self.latent_downsample // encoder_factor
        if not _is_power_of_two(remaining):
            raise ValueError("Remaining bottleneck downsample must be a power of two.")

        layers: list[nn.Module] = []
        for _ in range(int(math.log2(remaining))):
            layers.extend(
                [
                    nn.Conv2d(
                        encoder_out_channels,
                        encoder_out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GELU(),
                ]
            )
        self.spatial_downsample = nn.Sequential(*layers) if layers else nn.Identity()
        self.latent_projection = nn.Conv2d(
            encoder_out_channels, self.latent_dim, kernel_size=1
        )

        img_size = (
            int(cfg_get(params, "img_size_x")),
            int(cfg_get(params, "img_size_y")),
        )
        padded_size = (
            ceil_to_multiple(img_size[0], _input_alignment(params)),
            ceil_to_multiple(img_size[1], _input_alignment(params)),
        )
        latent_size = (
            padded_size[0] // self.latent_downsample,
            padded_size[1] // self.latent_downsample,
        )
        window_size = int(
            cfg_get(
                params,
                "bottleneck_window_size",
                min(4, int(cfg_get(params, "window_size", 8))),
            )
        )
        bottleneck_size = (
            ceil_to_multiple(latent_size[0], window_size),
            ceil_to_multiple(latent_size[1], window_size),
        )
        self.window_size = window_size
        self.longitude_periodic = bool(
            cfg_get(params, "encoder_longitude_periodic", False)
        )
        self.latitude_mode = str(
            cfg_get(params, "encoder_latitude_padding", "replicate")
        )
        self.swin_bottleneck = SwinFeatureStage(
            img_size=bottleneck_size,
            embed_dim=self.latent_dim,
            depths=_as_int_list(
                cfg_get(params, "bottleneck_depths", [1]), "bottleneck_depths"
            ),
            num_heads=_as_int_list(
                cfg_get(params, "bottleneck_num_heads", [4]),
                "bottleneck_num_heads",
            ),
            window_size=window_size,
            mlp_ratio=float(cfg_get(params, "mlp_ratio", 2.0)),
            qkv_bias=bool(cfg_get(params, "qkv_bias", True)),
            qk_scale=cfg_get(params, "qk_scale", None),
            drop_rate=float(cfg_get(params, "drop_rate", 0.0)),
            attn_drop_rate=float(cfg_get(params, "attn_drop_rate", 0.0)),
            drop_path_rate=float(cfg_get(params, "drop_path_rate", 0.0)),
            patch_size=1,
            patch_norm=bool(cfg_get(params, "patch_norm", True)),
            use_checkpoint=bool(cfg_get(params, "use_checkpoint", False)),
            resi_connection=str(cfg_get(params, "resi_connection", "1conv")),
            norm_layer=norm_layer,
        )
        self.branch_latent = BranchLatentModule(
            branch_channels=branch_channels,
            shared_name="shared",
            proj_alpha_init=float(cfg_get(params, "branch_proj_alpha_init", 0.0)),
            hub_beta_init=float(cfg_get(params, "hub_beta_init", 0.0)),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.latent_projection(self.spatial_downsample(x))
        zh, zw = z.shape[-2:]
        target_h = ceil_to_multiple(zh, self.window_size)
        target_w = ceil_to_multiple(zw, self.window_size)
        z = _pad_geophysical_field(
            z,
            target_h=target_h,
            target_w=target_w,
            longitude_periodic=self.longitude_periodic,
            latitude_mode=self.latitude_mode,
        )
        z = self.swin_bottleneck(z)
        return z[:, :, :zh, :zw]

    def split(self, z: Tensor) -> Dict[str, Tensor]:
        return self.branch_latent.split(z)

    def pack(self, branches: Mapping[str, Tensor]) -> Tensor:
        return self.branch_latent.pack(branches)

    def mix_branches(self, z: Tensor, return_before_hub: bool = False):
        return self.branch_latent(z, return_before_hub=return_before_hub)


class GeoConv2d(nn.Module):
    """2-D convolution with explicit latitude/longitude boundary semantics."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 1,
        bias: bool = True,
        longitude_periodic: bool = False,
        latitude_mode: str = "replicate",
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("GeoConv2d requires a positive odd kernel_size.")
        if latitude_mode not in {"reflect", "replicate"}:
            raise ValueError("latitude_mode must be 'reflect' or 'replicate'.")
        self.padding = kernel_size // 2
        self.longitude_periodic = bool(longitude_periodic)
        self.latitude_mode = str(latitude_mode)
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=kernel_size,
            padding=0,
            groups=int(groups),
            bias=bool(bias),
        )

    def forward(self, x: Tensor) -> Tensor:
        pad = self.padding
        if pad:
            longitude_mode = "circular" if self.longitude_periodic else "replicate"
            x = F.pad(x, (pad, pad, 0, 0), mode=longitude_mode)
            latitude_mode = self.latitude_mode
            if latitude_mode == "reflect" and pad >= x.shape[-2]:
                latitude_mode = "replicate"
            x = F.pad(x, (0, 0, pad, pad), mode=latitude_mode)
        return self.conv(x)


class GeoConvAct(nn.Module):
    """Boundary-aware convolution followed by GELU, optionally separable."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        longitude_periodic: bool,
        latitude_mode: str,
        separable: bool,
    ):
        super().__init__()
        in_channels, out_channels = int(in_channels), int(out_channels)
        if separable:
            self.net = nn.Sequential(
                GeoConv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    groups=in_channels,
                    longitude_periodic=longitude_periodic,
                    latitude_mode=latitude_mode,
                ),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GELU(),
            )
        else:
            self.net = nn.Sequential(
                GeoConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    longitude_periodic=longitude_periodic,
                    latitude_mode=latitude_mode,
                ),
                nn.GELU(),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GeoConvBlock(nn.Module):
    """Same-resolution local residual block used when a Swin stage is disabled."""

    def __init__(
        self,
        channels: int,
        longitude_periodic: bool,
        latitude_mode: str,
        separable: bool,
    ):
        super().__init__()
        channels = int(channels)
        self.body = nn.Sequential(
            GeoConvAct(
                channels,
                channels,
                longitude_periodic=longitude_periodic,
                latitude_mode=latitude_mode,
                separable=separable,
            ),
            GeoConv2d(
                channels,
                channels,
                kernel_size=3,
                groups=channels if separable else 1,
                longitude_periodic=longitude_periodic,
                latitude_mode=latitude_mode,
            ),
            nn.Conv2d(channels, channels, kernel_size=1)
            if separable
            else nn.Identity(),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.body(x))


class GeoUpsampleBlock(nn.Module):
    """Bilinear 2x upsampling followed by boundary-aware local reconstruction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        longitude_periodic: bool,
        latitude_mode: str,
        separable: bool,
    ):
        super().__init__()
        self.refine = GeoConvAct(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            longitude_periodic=longitude_periodic,
            latitude_mode=latitude_mode,
            separable=separable,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.refine(x)


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels for channel-first feature maps."""

    def __init__(self, channels: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channels = int(channels)
        self.norm = norm_layer(self.channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.channels:
            raise RuntimeError(
                f"ChannelLayerNorm2d expected channels={self.channels}, "
                f"got {x.shape[1]}."
            )
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class _LearnedUpsamplePostprocess(nn.Module):
    """Common normalization/activation used by learned upsampling ablations."""

    def __init__(self, channels: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.net = nn.Sequential(
            ChannelLayerNorm2d(channels, norm_layer=norm_layer),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvTransposeUpsampleBlock(nn.Module):
    """Boundary-aware 2x transposed-convolution decoder transition (U2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        longitude_periodic: bool,
        latitude_mode: str,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        if latitude_mode not in {"reflect", "replicate"}:
            raise ValueError("latitude_mode must be 'reflect' or 'replicate'.")
        self.longitude_periodic = bool(longitude_periodic)
        self.latitude_mode = str(latitude_mode)
        self.upsample = nn.ConvTranspose2d(
            int(in_channels),
            int(out_channels),
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.post = _LearnedUpsamplePostprocess(out_channels, norm_layer=norm_layer)

    def forward(self, x: Tensor) -> Tensor:
        # One-cell extension lets the transposed kernel see the requested
        # geophysical boundary. Cropping two output cells preserves exact 2x size.
        longitude_mode = "circular" if self.longitude_periodic else "replicate"
        x = F.pad(x, (1, 1, 0, 0), mode=longitude_mode)
        latitude_mode = self.latitude_mode
        if latitude_mode == "reflect" and x.shape[-2] <= 1:
            latitude_mode = "replicate"
        x = F.pad(x, (0, 0, 1, 1), mode=latitude_mode)
        x = self.upsample(x)
        x = x[:, :, 2:-2, 2:-2]
        return self.post(x)


class PixelShuffleUpsampleBlock(nn.Module):
    """Boundary-aware sub-pixel convolution decoder transition (U3)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        longitude_periodic: bool,
        latitude_mode: str,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.expand = GeoConv2d(
            int(in_channels),
            4 * int(out_channels),
            kernel_size=3,
            longitude_periodic=longitude_periodic,
            latitude_mode=latitude_mode,
        )
        self.shuffle = nn.PixelShuffle(upscale_factor=2)
        self.post = _LearnedUpsamplePostprocess(out_channels, norm_layer=norm_layer)

    def forward(self, x: Tensor) -> Tensor:
        return self.post(self.shuffle(self.expand(x)))


class LearnedPatchExpandUpsampleBlock(nn.Module):
    """Point-wise token expansion decoder transition with common postprocess (U4)."""

    def __init__(self, in_dim: int, out_dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.expand = nn.Linear(self.in_dim, 4 * self.out_dim, bias=False)
        self.post = _LearnedUpsamplePostprocess(self.out_dim, norm_layer=norm_layer)

    def forward(self, x: Tensor) -> Tensor:
        b, channels, h, w = x.shape
        if channels != self.in_dim:
            raise RuntimeError(
                f"LearnedPatchExpand expected channels={self.in_dim}, "
                f"got {channels}."
            )
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.expand(x)
        x = x.view(b, h, w, 2, 2, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(b, 2 * h, 2 * w, self.out_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.post(x)


class InterStagePatchExpanding(nn.Module):
    """Independent 2x token expansion used as the inverse of PatchMerging."""

    def __init__(self, in_dim: int, out_dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.expand = nn.Linear(self.in_dim, 4 * self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x: Tensor) -> Tensor:
        b, channels, h, w = x.shape
        if channels != self.in_dim:
            raise RuntimeError(
                f"PatchExpanding expected channels={self.in_dim}, got {channels}."
            )
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.expand(x)
        x = x.view(b, h, w, 2, 2, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(b, 2 * h, 2 * w, self.out_dim)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def _decoder_scale_plan(
    params, padded_size: Tuple[int, int]
) -> Tuple[list[Tuple[int, int]], list[str]]:
    latent_downsample = int(cfg_get(params, "latent_downsample", 4))
    stem_downsample = int(cfg_get(params, "stem_downsample", 1))
    encoder_merges = int(cfg_get(params, "encoder_merges", 0))
    encoder_factor = stem_downsample * (2**encoder_merges)
    if latent_downsample % encoder_factor:
        raise ValueError(
            f"latent_downsample={latent_downsample} must be divisible by "
            f"encoder factor={encoder_factor}."
        )
    remaining = latent_downsample // encoder_factor
    if not all(
        _is_power_of_two(value)
        for value in (latent_downsample, stem_downsample, remaining)
    ):
        raise ValueError("Decoder scale factors must be positive powers of two.")

    bottleneck_steps = int(math.log2(remaining))
    stem_steps = int(math.log2(stem_downsample))
    upsample_kinds = (
        ["bottleneck_geo"] * bottleneck_steps
        + ["patch_expand"] * encoder_merges
        + ["stem_geo"] * stem_steps
    )
    expected_steps = int(math.log2(latent_downsample))
    if len(upsample_kinds) != expected_steps:
        raise RuntimeError(
            f"Decoder scale plan has {len(upsample_kinds)} steps; "
            f"expected {expected_steps}."
        )

    latent_size = (
        padded_size[0] // latent_downsample,
        padded_size[1] // latent_downsample,
    )
    stage_sizes = [latent_size]
    for _ in upsample_kinds:
        previous = stage_sizes[-1]
        stage_sizes.append((2 * previous[0], 2 * previous[1]))
    if stage_sizes[-1] != tuple(padded_size):
        raise RuntimeError(
            f"Decoder final size={stage_sizes[-1]} does not match padded "
            f"input size={padded_size}."
        )
    return stage_sizes, upsample_kinds


def _default_decoder_dims(
    base_dim: int,
    stage_sizes: Sequence[Tuple[int, int]],
    reduce_above_pixels: int,
    min_dim: int,
) -> list[int]:
    dims = [int(base_dim)]
    current = int(base_dim)
    for h, w in stage_sizes[1:]:
        if h * w > int(reduce_above_pixels):
            current = max(int(min_dim), current // 2)
        dims.append(current)
    return dims


def _default_decoder_depths(
    stage_sizes: Sequence[Tuple[int, int]],
    swin_max_pixels: int,
    default_depth: int,
) -> list[int]:
    return [
        int(default_depth) if h * w <= int(swin_max_pixels) else 0
        for h, w in stage_sizes
    ]


class HierarchicalDecoderHead(nn.Module):
    """Scale-aware decoder with independent refinement and upsampling modules."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stage_sizes: Sequence[Tuple[int, int]],
        upsample_kinds: Sequence[str],
        stage_dims: Sequence[int],
        stage_depths: Sequence[int],
        stage_heads: Sequence[int],
        window_size: int,
        params,
        longitude_periodic: bool,
        latitude_mode: str,
        geo_separable: bool,
        output_kernel_size: int,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.stage_sizes = [tuple(map(int, size)) for size in stage_sizes]
        self.upsample_kinds = [str(kind) for kind in upsample_kinds]
        self.stage_dims = [int(value) for value in stage_dims]
        self.stage_depths = [int(value) for value in stage_depths]
        self.stage_heads = [int(value) for value in stage_heads]
        stage_count = len(self.stage_sizes)
        if not (
            len(self.stage_dims)
            == len(self.stage_depths)
            == len(self.stage_heads)
            == stage_count
        ):
            raise ValueError(
                "Decoder sizes, dims, depths, and heads must have equal length; "
                f"got {stage_count}, {len(self.stage_dims)}, "
                f"{len(self.stage_depths)}, {len(self.stage_heads)}."
            )
        if len(self.upsample_kinds) != stage_count - 1:
            raise ValueError("Decoder must have exactly one upsampler between stages.")
        if any(dim <= 0 for dim in self.stage_dims):
            raise ValueError("Decoder stage dimensions must be positive.")
        if any(depth < 0 for depth in self.stage_depths):
            raise ValueError("Decoder stage depths must be non-negative.")

        self.input_projection = nn.Sequential(
            nn.Conv2d(self.in_ch, self.stage_dims[0], kernel_size=1),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        for i, (size, dim, depth, heads) in enumerate(
            zip(
                self.stage_sizes,
                self.stage_dims,
                self.stage_depths,
                self.stage_heads,
            )
        ):
            if depth > 0:
                if heads <= 0 or dim % heads:
                    raise ValueError(
                        f"Decoder stage {i}: dim={dim} must be divisible by "
                        f"positive num_heads={heads}."
                    )
                if size[0] % window_size or size[1] % window_size:
                    raise ValueError(
                        f"Decoder stage {i}: size={size} must be divisible by "
                        f"window_size={window_size}."
                    )
                stage = SwinFeatureStage(
                    img_size=size,
                    embed_dim=dim,
                    depths=[depth],
                    num_heads=[heads],
                    window_size=int(window_size),
                    mlp_ratio=float(cfg_get(params, "mlp_ratio", 2.0)),
                    qkv_bias=bool(cfg_get(params, "qkv_bias", True)),
                    qk_scale=cfg_get(params, "qk_scale", None),
                    drop_rate=float(cfg_get(params, "drop_rate", 0.0)),
                    attn_drop_rate=float(cfg_get(params, "attn_drop_rate", 0.0)),
                    drop_path_rate=float(cfg_get(params, "drop_path_rate", 0.0)),
                    patch_size=1,
                    patch_norm=bool(cfg_get(params, "patch_norm", True)),
                    use_checkpoint=bool(cfg_get(params, "use_checkpoint", False)),
                    resi_connection=str(cfg_get(params, "resi_connection", "1conv")),
                    norm_layer=norm_layer,
                )
            else:
                stage = GeoConvBlock(
                    channels=dim,
                    longitude_periodic=longitude_periodic,
                    latitude_mode=latitude_mode,
                    separable=geo_separable,
                )
            self.stages.append(stage)

        self.upsamples = nn.ModuleList()
        for i, kind in enumerate(self.upsample_kinds):
            in_dim, out_dim = self.stage_dims[i], self.stage_dims[i + 1]
            if kind == "patch_expand":
                upsample = InterStagePatchExpanding(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    norm_layer=norm_layer,
                )
            elif kind in {"bottleneck_geo", "stem_geo"}:
                upsample = GeoUpsampleBlock(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    longitude_periodic=longitude_periodic,
                    latitude_mode=latitude_mode,
                    separable=geo_separable,
                )
            elif kind == "conv_transpose":
                upsample = ConvTransposeUpsampleBlock(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    longitude_periodic=longitude_periodic,
                    latitude_mode=latitude_mode,
                    norm_layer=norm_layer,
                )
            elif kind == "pixel_shuffle":
                upsample = PixelShuffleUpsampleBlock(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    longitude_periodic=longitude_periodic,
                    latitude_mode=latitude_mode,
                    norm_layer=norm_layer,
                )
            elif kind == "learned_patch_expand":
                upsample = LearnedPatchExpandUpsampleBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    norm_layer=norm_layer,
                )
            else:
                raise ValueError(f"Unknown decoder upsample kind {kind!r}.")
            self.upsamples.append(upsample)

        output_kernel_size = int(output_kernel_size)
        if output_kernel_size == 1:
            self.output_projection = nn.Conv2d(
                self.stage_dims[-1], self.out_ch, kernel_size=1
            )
        elif output_kernel_size == 3:
            self.output_projection = GeoConv2d(
                self.stage_dims[-1],
                self.out_ch,
                kernel_size=3,
                longitude_periodic=longitude_periodic,
                latitude_mode=latitude_mode,
            )
        else:
            raise ValueError("decoder_output_kernel_size must be 1 or 3.")

    def forward(
        self, z: Tensor, output_size: Optional[Tuple[int, int]] = None
    ) -> Tensor:
        if z.shape[1] != self.in_ch:
            raise RuntimeError(
                f"Decoder expected input channels={self.in_ch}, got {z.shape[1]}."
            )
        if tuple(z.shape[-2:]) != self.stage_sizes[0]:
            raise RuntimeError(
                f"Decoder expected latent size={self.stage_sizes[0]}, "
                f"got {tuple(z.shape[-2:])}."
            )
        x = self.input_projection(z)
        for i, stage in enumerate(self.stages):
            expected_size = self.stage_sizes[i]
            expected_dim = self.stage_dims[i]
            if tuple(x.shape[-2:]) != expected_size or x.shape[1] != expected_dim:
                raise RuntimeError(
                    f"Decoder stage {i} state mismatch: x={tuple(x.shape)}, "
                    f"expected channels/size={expected_dim}/{expected_size}."
                )
            x = stage(x)
            if i < len(self.upsamples):
                x = self.upsamples[i](x)
        x = self.output_projection(x)
        if output_size is not None:
            h, w = map(int, output_size)
            x = x[:, :, :h, :w]
        return x


class PhySP_AEv3(nn.Module):
    """Semantic, hierarchical, region-to-global physical-space autoencoder."""

    DEFAULT_GROUP_INDICES = StructuredSwinAE.DEFAULT_GROUP_INDICES
    DEFAULT_STEM_CHANNELS = StructuredSwinAE.DEFAULT_STEM_CHANNELS
    DEFAULT_BRANCH_CHANNELS_64 = StructuredSwinAE.DEFAULT_BRANCH_CHANNELS_64
    DEFAULT_BRANCH_CHANNELS_96 = StructuredSwinAE.DEFAULT_BRANCH_CHANNELS_96

    def __init__(self, params, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_chans = int(cfg_get(params, "in_chans"))
        self.out_chans = int(cfg_get(params, "out_chans"))
        self.img_range = float(cfg_get(params, "img_range", 1.0))
        self.window_size = int(cfg_get(params, "window_size", 8))
        self.embed_dim = int(cfg_get(params, "embed_dim", 64))
        self.latent_dim = int(cfg_get(params, "latent_dim", 64))
        self.latent_downsample = int(cfg_get(params, "latent_downsample", 4))
        self.img_size = (
            int(cfg_get(params, "img_size_x")),
            int(cfg_get(params, "img_size_y")),
        )
        self.input_alignment = _input_alignment(params)
        self.padded_img_size = (
            ceil_to_multiple(self.img_size[0], self.input_alignment),
            ceil_to_multiple(self.img_size[1], self.input_alignment),
        )
        self.longitude_periodic = bool(
            cfg_get(params, "encoder_longitude_periodic", False)
        )
        self.latitude_mode = str(
            cfg_get(params, "encoder_latitude_padding", "replicate")
        )

        group_indices = cfg_get(params, "group_indices", None)
        if group_indices is None:
            group_indices = self.DEFAULT_GROUP_INDICES
        self.group_indices = OrderedDict((k, list(v)) for k, v in group_indices.items())

        stem_channels = cfg_get(params, "stem_channels", None)
        if stem_channels is None:
            stem_channels = self.DEFAULT_STEM_CHANNELS
        self.stem_channels = OrderedDict(
            (name, int(stem_channels[name])) for name in self.group_indices
        )

        branch_channels = cfg_get(params, "branch_channels", None)
        if branch_channels is None:
            branch_channels = (
                self.DEFAULT_BRANCH_CHANNELS_96
                if self.latent_dim == 96
                else self.DEFAULT_BRANCH_CHANNELS_64
            )
        self.branch_channels = OrderedDict(
            (name, int(value)) for name, value in branch_channels.items()
        )
        if sum(self.branch_channels.values()) != self.latent_dim:
            raise ValueError(
                f"sum(branch_channels)={sum(self.branch_channels.values())} "
                f"must equal latent_dim={self.latent_dim}."
            )
        for name in self.group_indices:
            if name not in self.branch_channels:
                raise ValueError(f"Missing branch channel for group {name!r}.")

        self.register_buffer("mean", torch.zeros(1, 1, 1, 1), persistent=False)
        self.group_stems = GroupStemBank(self.group_indices, self.stem_channels)
        self.shared_encoder = SharedHierarchicalSwinEncoder(
            params=params,
            in_ch=self.group_stems.out_channels,
            norm_layer=norm_layer,
        )
        self.bottleneck = HierarchicalStructuredBottleneck(
            params=params,
            encoder_out_channels=self.shared_encoder.out_channels,
            branch_channels=self.branch_channels,
            norm_layer=norm_layer,
        )
        self.branch_latent = self.bottleneck.branch_latent

        self.decoder_stage_sizes, self.decoder_upsample_kinds = _decoder_scale_plan(
            params=params,
            padded_size=self.padded_img_size,
        )
        decoder_upsample_type = str(
            cfg_get(params, "decoder_upsample_type", "auto")
        ).lower()
        upsample_aliases = {
            "auto": None,
            "geo": None,
            "conv_transpose": "conv_transpose",
            "deconv": "conv_transpose",
            "pixel_shuffle": "pixel_shuffle",
            "pixelshuffle": "pixel_shuffle",
            "patch_expand": "learned_patch_expand",
            "learned_patch_expand": "learned_patch_expand",
        }
        if decoder_upsample_type not in upsample_aliases:
            raise ValueError(
                "decoder_upsample_type must be one of auto, conv_transpose, "
                "pixel_shuffle, or patch_expand; "
                f"got {decoder_upsample_type!r}."
            )
        override_kind = upsample_aliases[decoder_upsample_type]
        if override_kind is not None:
            self.decoder_upsample_kinds = [
                override_kind for _ in self.decoder_upsample_kinds
            ]
        decoder_stage_count = len(self.decoder_stage_sizes)
        decoder_window_size = int(
            cfg_get(
                params,
                "decoder_window_size",
                cfg_get(params, "group_refine_window_size", self.window_size),
            )
        )
        decoder_longitude_periodic = bool(
            cfg_get(
                params,
                "decoder_longitude_periodic",
                self.longitude_periodic,
            )
        )
        decoder_latitude_mode = str(
            cfg_get(params, "decoder_latitude_padding", self.latitude_mode)
        )
        if decoder_latitude_mode not in {"reflect", "replicate"}:
            raise ValueError(
                "decoder_latitude_padding must be 'reflect' or 'replicate'."
            )
        large_grid = self.padded_img_size[0] * self.padded_img_size[1] > 57600
        geo_separable_cfg = cfg_get(params, "decoder_geo_separable", None)
        decoder_geo_separable = (
            large_grid if geo_separable_cfg is None else bool(geo_separable_cfg)
        )
        output_kernel_cfg = cfg_get(params, "decoder_output_kernel_size", None)
        decoder_output_kernel_size = (
            1 if output_kernel_cfg is None and large_grid else int(output_kernel_cfg or 3)
        )

        shared_default_dims = _default_decoder_dims(
            base_dim=int(cfg_get(params, "shared_decoder_embed_dim", self.embed_dim)),
            stage_sizes=self.decoder_stage_sizes,
            reduce_above_pixels=int(
                cfg_get(params, "shared_decoder_reduce_above_pixels", 57600)
            ),
            min_dim=int(cfg_get(params, "shared_decoder_min_dim", 16)),
        )
        shared_dims_value = cfg_get(params, "shared_decoder_dims", None)
        shared_dims = _as_int_list(
            shared_default_dims if shared_dims_value is None else shared_dims_value,
            "shared_decoder_dims",
        )
        shared_default_depths = _default_decoder_depths(
            stage_sizes=self.decoder_stage_sizes,
            swin_max_pixels=int(
                cfg_get(params, "shared_decoder_swin_max_pixels", 57600)
            ),
            default_depth=int(cfg_get(params, "shared_decoder_default_depth", 1)),
        )
        shared_depths_value = cfg_get(params, "shared_decoder_depths", None)
        shared_depths = _as_nonnegative_int_list(
            shared_default_depths
            if shared_depths_value is None
            else shared_depths_value,
            "shared_decoder_depths",
        )
        shared_heads_value = cfg_get(params, "shared_decoder_num_heads", None)
        if shared_heads_value is None:
            preferred_heads = int(
                cfg_get(params, "shared_decoder_default_num_heads", 4)
            )
            shared_heads = [
                _compatible_num_heads(dim, preferred_heads) for dim in shared_dims
            ]
        else:
            shared_heads = _as_int_list(
                shared_heads_value, "shared_decoder_num_heads"
            )
        for name, values in {
            "shared_decoder_dims": shared_dims,
            "shared_decoder_depths": shared_depths,
            "shared_decoder_num_heads": shared_heads,
        }.items():
            if len(values) != decoder_stage_count:
                raise ValueError(
                    f"{name} must have {decoder_stage_count} values for sizes "
                    f"{self.decoder_stage_sizes}; got {values}."
                )

        shared_ch = self.branch_channels["shared"]
        self.shared_base_decoder = HierarchicalDecoderHead(
            in_ch=shared_ch,
            out_ch=self.out_chans,
            stage_sizes=self.decoder_stage_sizes,
            upsample_kinds=self.decoder_upsample_kinds,
            stage_dims=shared_dims,
            stage_depths=shared_depths,
            stage_heads=shared_heads,
            window_size=decoder_window_size,
            params=params,
            longitude_periodic=decoder_longitude_periodic,
            latitude_mode=decoder_latitude_mode,
            geo_separable=decoder_geo_separable,
            output_kernel_size=decoder_output_kernel_size,
            norm_layer=norm_layer,
        )

        group_dims_cfg = cfg_get(params, "group_decoder_dims", None)
        group_depths_cfg = cfg_get(params, "group_decoder_depths", None)
        group_heads_cfg = cfg_get(params, "group_decoder_num_heads", None)
        group_base_dim = int(
            cfg_get(
                params,
                "group_decoder_embed_dim",
                cfg_get(params, "group_refine_embed_dim", self.embed_dim),
            )
        )
        group_default_dims = _default_decoder_dims(
            base_dim=group_base_dim,
            stage_sizes=self.decoder_stage_sizes,
            reduce_above_pixels=int(
                cfg_get(params, "group_decoder_reduce_above_pixels", 14400)
            ),
            min_dim=int(cfg_get(params, "group_decoder_min_dim", 8)),
        )
        group_default_depths = _default_decoder_depths(
            stage_sizes=self.decoder_stage_sizes,
            swin_max_pixels=int(
                cfg_get(params, "group_decoder_swin_max_pixels", 14400)
            ),
            default_depth=int(
                cfg_get(
                    params,
                    "group_decoder_default_depth",
                    cfg_get(params, "group_refine_depth", 1),
                )
            ),
        )
        group_default_heads = int(
            cfg_get(
                params,
                "group_decoder_default_num_heads",
                cfg_get(params, "group_refine_default_num_heads", 2),
            )
        )
        decoders = OrderedDict()
        for name, indices in self.group_indices.items():
            dims_value = _resolve_group_config(
                group_dims_cfg, name, group_default_dims
            )
            depths_value = _resolve_group_config(
                group_depths_cfg, name, group_default_depths
            )
            heads_value = _resolve_group_config(group_heads_cfg, name, None)
            group_dims = _as_int_list(dims_value, f"group_decoder_dims[{name}]")
            group_depths = _as_nonnegative_int_list(
                depths_value, f"group_decoder_depths[{name}]"
            )
            if heads_value is None:
                group_heads = [
                    _compatible_num_heads(dim, group_default_heads)
                    for dim in group_dims
                ]
            else:
                group_heads = _as_int_list(
                    heads_value, f"group_decoder_num_heads[{name}]"
                )
            for config_name, values in {
                "dims": group_dims,
                "depths": group_depths,
                "num_heads": group_heads,
            }.items():
                if len(values) != decoder_stage_count:
                    raise ValueError(
                        f"group_decoder_{config_name}[{name}] must have "
                        f"{decoder_stage_count} values; got {values}."
                    )
            decoders[name] = HierarchicalDecoderHead(
                in_ch=shared_ch + self.branch_channels[name],
                out_ch=len(indices),
                stage_sizes=self.decoder_stage_sizes,
                upsample_kinds=self.decoder_upsample_kinds,
                stage_dims=group_dims,
                stage_depths=group_depths,
                stage_heads=group_heads,
                window_size=decoder_window_size,
                params=params,
                longitude_periodic=decoder_longitude_periodic,
                latitude_mode=decoder_latitude_mode,
                geo_separable=decoder_geo_separable,
                output_kernel_size=decoder_output_kernel_size,
                norm_layer=norm_layer,
            )
        self.group_residual_decoders = nn.ModuleDict(decoders)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _pad_input(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        target_h = ceil_to_multiple(h, self.input_alignment)
        target_w = ceil_to_multiple(w, self.input_alignment)
        return _pad_geophysical_field(
            x,
            target_h=target_h,
            target_w=target_w,
            longitude_periodic=self.longitude_periodic,
            latitude_mode=self.latitude_mode,
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        h, w = x.shape[-2:]
        x = (x - self.mean.to(dtype=x.dtype, device=x.device)) * self.img_range
        x = self._pad_input(x)
        stem_features = self.group_stems(x)
        grouped = self.group_stems.concat(stem_features)
        encoded = self.shared_encoder(grouped)
        return self.bottleneck(encoded), (h, w)

    def encode_det(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        return self.encode(x)

    def decode(
        self,
        z: Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        return_parts: bool = False,
    ):
        branches_mixed, branches_before_hub = self.bottleneck.mix_branches(
            z, return_before_hub=True
        )
        z_shared = branches_mixed["shared"]
        x_base_low = self.shared_base_decoder(z_shared, output_size=output_size)
        x_hat = x_base_low.clone()
        residuals = OrderedDict()
        for name, indices in self.group_indices.items():
            z_group = branches_mixed[name]
            z_in = torch.cat((z_shared, z_group), dim=1)
            residual = self.group_residual_decoders[name](
                z_in, output_size=output_size
            )
            residuals[name] = residual
            x_hat[:, indices, :, :] = x_base_low[:, indices, :, :] + residual

        mean = self.mean.to(dtype=x_hat.dtype, device=x_hat.device)
        x_hat = x_hat / self.img_range + mean
        x_base_low_out = x_base_low / self.img_range + mean
        if not return_parts:
            return x_hat
        return {
            "x_hat": x_hat,
            "x_base_low": x_base_low_out,
            "residuals": residuals,
            "branches": branches_mixed,
            "branches_before_hub": branches_before_hub,
            "branch_raw": branches_before_hub,
            "z_raw": z,
        }

    def decode_det(
        self,
        z: Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        return_parts: bool = False,
    ):
        return self.decode(z, output_size=output_size, return_parts=return_parts)

    def forward(
        self,
        x: Tensor,
        return_latent: bool = False,
        return_parts: bool = False,
    ):
        z, output_size = self.encode(x)
        output = self.decode(z, output_size=output_size, return_parts=return_parts)
        if return_parts:
            output["output_size"] = output_size
            if return_latent:
                output["z"] = z
            return output
        if return_latent:
            return output, z
        return output

    def freeze_ae(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    @torch.no_grad()
    def split_latent(self, z: Tensor) -> Dict[str, Tensor]:
        return self.bottleneck.split(z)

    def pack_branches(self, branches: Mapping[str, Tensor]) -> Tensor:
        return self.bottleneck.pack(branches)

    @staticmethod
    def low_frequency_target(
        x: Tensor,
        kernel_size: int = 5,
        lowpass_type: str = "gaussian",
        sigma: float = 1.0,
    ) -> Tensor:
        return StructuredSwinAE.low_frequency_target(
            x,
            kernel_size=kernel_size,
            lowpass_type=lowpass_type,
            sigma=sigma,
        )


# Final gated Shared-Private model (archived V4 implementation).


import math
from collections import OrderedDict
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn


Tensor = torch.Tensor


def _resolve_branch_value(value, branch_name: str, default: float) -> float:
    """Resolve a scalar or branch-name mapping from the model configuration."""
    if isinstance(value, Mapping):
        return float(value.get(branch_name, value.get("default", default)))
    return float(value)


class IndependentGatedSharedHubMixer(nn.Module):
    """Shared/private hub with one learnable private-to-shared gate per branch.

    V3 uses one global scalar ``beta`` after concatenating all private summaries.
    V4 instead scales each private summary independently before the common hub:

        shared_new = shared + F(shared, gate_1*m_1, ..., gate_n*m_n)
                            - F(shared, 0, ..., 0)

    The null-private response is subtracted so that all-zero gates produce an
    exact identity update for the shared branch. The common nonlinear hub and
    branch-specific FiLM paths are retained from V3.

    Gates use a bounded signed tanh parameterization. The 1x1 projections learn
    response sign and structure; the gates provide an interpretable coupling
    strength and cannot grow without bound.
    """

    def __init__(
        self,
        branch_channels: Mapping[str, int],
        shared_name: str = "shared",
        hidden_ch: Optional[int] = None,
        gate_init=0.0,
        gate_max: float = 1.0,
        gate_trainable: bool = True,
    ):
        super().__init__()
        self.branch_channels = OrderedDict(
            (name, int(channels)) for name, channels in branch_channels.items()
        )
        self.shared_name = str(shared_name)
        if self.shared_name not in self.branch_channels:
            raise ValueError(
                f"shared_name={self.shared_name!r} not found in branch_channels"
            )

        self.private_names = [
            name for name in self.branch_channels if name != self.shared_name
        ]
        if not self.private_names:
            raise ValueError("At least one private branch is required.")

        self.shared_ch = self.branch_channels[self.shared_name]
        hidden_ch = int(hidden_ch or self.shared_ch)
        self.gate_max = float(gate_max)
        if not math.isfinite(self.gate_max) or self.gate_max <= 0:
            raise ValueError("hub_gate_max must be a finite positive scalar.")

        self.private_to_shared = nn.ModuleDict(
            {
                name: nn.Conv2d(
                    self.branch_channels[name],
                    self.shared_ch,
                    kernel_size=1,
                )
                for name in self.private_names
            }
        )

        hub_in_ch = self.shared_ch * (1 + len(self.private_names))
        self.shared_update = nn.Sequential(
            nn.Conv2d(hub_in_ch, hidden_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, self.shared_ch, kernel_size=1),
        )

        self.raw_gates = nn.ParameterDict()
        for name in self.private_names:
            initial_gate = _resolve_branch_value(gate_init, name, 0.0)
            raw_value = self._gate_to_raw(initial_gate)
            parameter = nn.Parameter(
                torch.tensor(raw_value, dtype=torch.get_default_dtype()),
                requires_grad=bool(gate_trainable),
            )
            self.raw_gates[name] = parameter

        # Shared -> private FiLM is unchanged from V3.
        self.film = nn.ModuleDict()
        for name in self.private_names:
            channels = self.branch_channels[name]
            mid = max(channels, self.shared_ch)
            modulator = nn.Sequential(
                nn.Conv2d(self.shared_ch, mid, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(mid, 2 * channels, kernel_size=1),
            )
            nn.init.zeros_(modulator[-1].weight)
            nn.init.zeros_(modulator[-1].bias)
            self.film[name] = modulator

    def _gate_to_raw(self, gate: float) -> float:
        """Convert an actual bounded gate value to its unconstrained parameter."""
        gate = float(gate)
        if not math.isfinite(gate):
            raise ValueError("Hub gate initial values must be finite.")
        ratio = gate / self.gate_max
        eps = 1e-6
        ratio = max(-1.0 + eps, min(1.0 - eps, ratio))
        return math.atanh(ratio)

    def gate_values(
        self,
        branch_mask: Optional[Mapping[str, float]] = None,
    ) -> "OrderedDict[str, Tensor]":
        """Return bounded gate values, optionally multiplied by a runtime mask."""
        values: "OrderedDict[str, Tensor]" = OrderedDict()
        for name in self.private_names:
            gate = self.gate_max * torch.tanh(self.raw_gates[name])
            if branch_mask is not None:
                gate = gate * float(branch_mask.get(name, 1.0))
            values[name] = gate
        return values

    @property
    def beta(self) -> Tensor:
        """Backward-compatible diagnostic view of the mean branch gate.

        V4 does not optimize this property. New code should use ``gate_values``.
        """
        return torch.stack(list(self.gate_values().values())).mean()

    def forward(
        self,
        branches: Mapping[str, Tensor],
        branch_mask: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Tensor]:
        shared = branches[self.shared_name]
        gates = self.gate_values(branch_mask=branch_mask)

        gated_summaries = []
        null_summaries = []
        for name in self.private_names:
            summary = self.private_to_shared[name](branches[name])
            gated_summaries.append(gates[name] * summary)
            null_summaries.append(torch.zeros_like(summary))

        hub_input = torch.cat([shared, *gated_summaries], dim=1)
        null_input = torch.cat([shared, *null_summaries], dim=1)
        private_conditioned_update = (
            self.shared_update(hub_input) - self.shared_update(null_input)
        )
        shared_new = shared + private_conditioned_update

        out: "OrderedDict[str, Tensor]" = OrderedDict()
        out[self.shared_name] = shared_new
        for name in self.private_names:
            private = branches[name]
            gamma_shift = self.film[name](shared_new)
            gamma, shift = torch.chunk(gamma_shift, chunks=2, dim=1)
            out[name] = private * (1.0 + gamma) + shift
        return out

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Upgrade a V3 global ``beta`` checkpoint to equal V4 branch gates."""
        legacy_beta_key = prefix + "beta"
        if legacy_beta_key in state_dict:
            legacy_beta = float(state_dict.pop(legacy_beta_key).detach().cpu())
            raw_value = self._gate_to_raw(legacy_beta)
            for name in self.private_names:
                gate_key = prefix + f"raw_gates.{name}"
                if gate_key not in state_dict:
                    state_dict[gate_key] = self.raw_gates[name].detach().new_tensor(
                        raw_value
                    )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self) -> str:
        return (
            f"shared_name={self.shared_name!r}, "
            f"private_names={self.private_names}, gate_max={self.gate_max}"
        )


class PhySP_AEv4(PhySP_AEv3):
    """PhySP-AE V4: V3 hierarchy and decoders with branch-independent hub gates."""

    def __init__(self, params, norm_layer=nn.LayerNorm):
        super().__init__(params=params, norm_layer=norm_layer)

        old_hub = self.bottleneck.branch_latent.hub
        gate_init = cfg_get(
            params,
            "hub_gate_init",
            cfg_get(params, "hub_beta_init", 0.0),
        )
        gate_max = float(cfg_get(params, "hub_gate_max", 1.0))
        gate_trainable = bool(cfg_get(params, "hub_gate_trainable", True))

        new_hub = IndependentGatedSharedHubMixer(
            branch_channels=self.branch_channels,
            shared_name="shared",
            gate_init=gate_init,
            gate_max=gate_max,
            gate_trainable=gate_trainable,
        )

        # Preserve V3 initialization for all unchanged Hub and FiLM tensors.
        transferable = OrderedDict(
            (key, value)
            for key, value in old_hub.state_dict().items()
            if key != "beta"
        )
        incompatible = new_hub.load_state_dict(transferable, strict=False)
        expected_missing = {
            f"raw_gates.{name}" for name in new_hub.private_names
        }
        unexpected_missing = set(incompatible.missing_keys) - expected_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected V3->V4 Hub initialization mismatch: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )

        self.bottleneck.branch_latent.hub = new_hub
        self.branch_latent = self.bottleneck.branch_latent

    def hub_gate_values(
        self,
        detach: bool = False,
    ) -> "OrderedDict[str, Tensor]":
        values = self.bottleneck.branch_latent.hub.gate_values()
        if detach:
            return OrderedDict(
                (name, value.detach()) for name, value in values.items()
            )
        return values

    def decode(
        self,
        z: Tensor,
        output_size=None,
        return_parts: bool = False,
    ):
        output = super().decode(
            z=z,
            output_size=output_size,
            return_parts=return_parts,
        )
        if return_parts:
            output["hub_gates"] = self.hub_gate_values(detach=False)
        return output


# Support both the historical ``AEvN`` class style and the requested file name.
PhySP_AEV4 = PhySP_AEv4
PhySP_AE = PhySP_AEv4

__all__ = [
    "IndependentGatedSharedHubMixer",
    "PhySP_AEv4",
    "PhySP_AEV4",
    "PhySP_AE",
]

OCSLDAAutoencoder = PhySP_AEv4
__all__.append("OCSLDAAutoencoder")
