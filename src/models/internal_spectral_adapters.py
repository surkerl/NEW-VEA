from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import open_clip


GRID_SIZE = 14
PATCH_TOKEN_COUNT = GRID_SIZE * GRID_SIZE


def _run_block(block: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    try:
        return block(tokens, attn_mask=None)
    except TypeError:
        return block(tokens)


class HaarDWT2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1) / 2.0
        self.register_buffer("filters", filters)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        channels = x.shape[1]
        weight = self.filters.to(dtype=x.dtype).repeat(channels, 1, 1, 1)
        bands = F.conv2d(x, weight, stride=2, groups=channels)
        bands = bands.reshape(x.shape[0], channels, 4, x.shape[2] // 2, x.shape[3] // 2)
        return tuple(bands.unbind(dim=2))  # type: ignore[return-value]


class HaarIDWT2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1) / 2.0
        self.register_buffer("filters", filters)

    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = ll.shape
        bands = torch.stack([ll, lh, hl, hh], dim=2).reshape(batch_size, 4 * channels, height, width)
        weight = self.filters.to(dtype=ll.dtype).repeat(channels, 1, 1, 1)
        return F.conv_transpose2d(bands, weight, stride=2, groups=channels)


class _TokenAdapterBase(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 128,
        adapter_dropout: float = 0.1,
        layer_scale_init: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.down_projection = nn.Linear(self.embed_dim, self.bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(adapter_dropout)
        self.up_projection = nn.Linear(self.bottleneck_dim, self.embed_dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(self.embed_dim))
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.normal_(self.up_projection.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.up_projection.bias)

    def _mix(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def forward(
        self,
        patch_tokens: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if patch_tokens.ndim != 3 or patch_tokens.shape[1] != PATCH_TOKEN_COUNT:
            raise ValueError(
                f"Expected patch tokens [B, {PATCH_TOKEN_COUNT}, D], found {tuple(patch_tokens.shape)}."
            )
        batch_size = patch_tokens.shape[0]
        reduced = self.down_projection(self.norm(patch_tokens))
        feature_map = reduced.transpose(1, 2).reshape(batch_size, self.bottleneck_dim, GRID_SIZE, GRID_SIZE)
        mixed, mixer_diagnostics = self._mix(feature_map)
        mixed_tokens = mixed.flatten(2).transpose(1, 2)
        residual = self.up_projection(self.dropout(self.activation(mixed_tokens)))
        adapted = patch_tokens + self.gamma.view(1, 1, -1) * residual
        diagnostics = {
            "residual_abs_mean": residual.abs().mean(dim=(1, 2)),
            "residual_norm": residual.norm(dim=-1).mean(dim=1),
            "layer_scale_mean": self.gamma.abs().mean(),
            **mixer_diagnostics,
        }
        if return_diagnostics:
            return adapted, diagnostics
        return adapted


class SpatialTokenAdapter(_TokenAdapterBase):
    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 128,
        adapter_dropout: float = 0.1,
        layer_scale_init: float = 1.0e-4,
    ) -> None:
        super().__init__(embed_dim, bottleneck_dim, adapter_dropout, layer_scale_init)
        self.depthwise_conv = nn.Conv2d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=3,
            padding=1,
            groups=bottleneck_dim,
            bias=True,
        )
        self.pointwise_conv = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=1)

    def _mix(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.pointwise_conv(F.gelu(self.depthwise_conv(x))), {}


class GlobalFilterTokenAdapter(_TokenAdapterBase):
    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 128,
        adapter_dropout: float = 0.1,
        layer_scale_init: float = 1.0e-4,
    ) -> None:
        super().__init__(embed_dim, bottleneck_dim, adapter_dropout, layer_scale_init)
        self.complex_delta = nn.Parameter(torch.zeros(GRID_SIZE, GRID_SIZE // 2 + 1, bottleneck_dim, 2))
        nn.init.normal_(self.complex_delta, mean=0.0, std=0.02)

        fy = torch.fft.fftfreq(GRID_SIZE).view(GRID_SIZE, 1)
        fx = torch.fft.rfftfreq(GRID_SIZE).view(1, GRID_SIZE // 2 + 1)
        radius = torch.sqrt(fy.square() + fx.square())
        radius = radius / radius.max().clamp_min(1.0e-6)
        self.register_buffer("low_mask", radius <= (1.0 / 3.0), persistent=False)
        self.register_buffer(
            "mid_mask",
            (radius > (1.0 / 3.0)) & (radius <= (2.0 / 3.0)),
            persistent=False,
        )
        self.register_buffer("high_mask", radius > (2.0 / 3.0), persistent=False)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = values[mask]
        return selected.mean() if selected.numel() else values.new_zeros(())

    def _mix(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        input_dtype = x.dtype
        x_grid = x.permute(0, 2, 3, 1)
        spectrum = torch.fft.rfft2(x_grid.float(), dim=(1, 2), norm="ortho")
        complex_weight = torch.view_as_complex(self.complex_delta.contiguous())
        filtered = spectrum * complex_weight.unsqueeze(0)
        residual = torch.fft.irfft2(
            filtered,
            s=(GRID_SIZE, GRID_SIZE),
            dim=(1, 2),
            norm="ortho",
        )
        magnitude = complex_weight.abs()
        diagnostics = {
            "global_filter_abs_mean": magnitude.mean(),
            "global_filter_low_mean": self._masked_mean(magnitude, self.low_mask),
            "global_filter_mid_mean": self._masked_mean(magnitude, self.mid_mask),
            "global_filter_high_mean": self._masked_mean(magnitude, self.high_mask),
        }
        return residual.permute(0, 3, 1, 2).to(dtype=input_dtype), diagnostics


class FactorizedSpectralTokenAdapter(_TokenAdapterBase):
    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 128,
        adapter_dropout: float = 0.1,
        layer_scale_init: float = 1.0e-4,
        radial_bands: int = 6,
        orientation_bins: int = 6,
    ) -> None:
        super().__init__(embed_dim, bottleneck_dim, adapter_dropout, layer_scale_init)
        self.radial_bands = int(radial_bands)
        self.orientation_bins = int(orientation_bins)
        self.basis_coeff = nn.Parameter(
            torch.zeros(self.radial_bands, self.orientation_bins, self.bottleneck_dim)
        )
        self.channel_scale = nn.Parameter(torch.ones(self.bottleneck_dim))
        self.register_buffer("spectral_basis", self._build_spectral_basis())

    def _build_spectral_basis(self) -> torch.Tensor:
        eps = 1.0e-6
        fy = torch.fft.fftfreq(GRID_SIZE)
        fx = torch.fft.rfftfreq(GRID_SIZE)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(fx_grid.square() + fy_grid.square())
        radius_position = radius / radius.max().clamp_min(eps) * max(self.radial_bands - 1, 1)
        radial_centers = torch.arange(self.radial_bands, dtype=torch.float32).view(-1, 1, 1)
        radial_basis = (1.0 - (radius_position.unsqueeze(0) - radial_centers).abs()).clamp_min(0.0)

        orientation = torch.remainder(torch.atan2(fy_grid, fx_grid + eps), math.pi)
        orientation_position = orientation / math.pi * self.orientation_bins
        orientation_centers = torch.arange(self.orientation_bins, dtype=torch.float32).view(-1, 1, 1)
        orientation_distance = (orientation_position.unsqueeze(0) - orientation_centers).abs()
        orientation_distance = torch.minimum(
            orientation_distance,
            self.orientation_bins - orientation_distance,
        )
        orientation_basis = (1.0 - orientation_distance).clamp_min(0.0)

        basis = radial_basis[:, None] * orientation_basis[None, :]
        return basis / basis.sum(dim=(0, 1), keepdim=True).clamp_min(eps)

    def _mix(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        input_dtype = x.dtype
        x_grid = x.permute(0, 2, 3, 1)
        spectrum = torch.fft.rfft2(x_grid.float(), dim=(1, 2), norm="ortho")
        gain = torch.einsum("kor,koij->ijr", self.basis_coeff, self.spectral_basis)
        gain = 0.1 * torch.tanh(gain) * self.channel_scale.view(1, 1, -1)
        filtered = spectrum * gain.unsqueeze(0)
        residual = torch.fft.irfft2(
            filtered,
            s=(GRID_SIZE, GRID_SIZE),
            dim=(1, 2),
            norm="ortho",
        )
        abs_coeff = self.basis_coeff.abs()
        diagnostics = {
            "factorized_coeff_abs_mean": abs_coeff.mean(),
            "factorized_radial_abs_mean": abs_coeff.mean(dim=(1, 2)),
            "factorized_orientation_abs_mean": abs_coeff.mean(dim=(0, 2)),
        }
        return residual.permute(0, 3, 1, 2).to(dtype=input_dtype), diagnostics


class WaveletTokenAdapter(_TokenAdapterBase):
    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 128,
        adapter_dropout: float = 0.1,
        layer_scale_init: float = 1.0e-4,
    ) -> None:
        super().__init__(embed_dim, bottleneck_dim, adapter_dropout, layer_scale_init)
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.ll_conv = self._depthwise_conv(bottleneck_dim)
        self.lh_conv = self._depthwise_conv(bottleneck_dim)
        self.hl_conv = self._depthwise_conv(bottleneck_dim)
        self.hh_conv = self._depthwise_conv(bottleneck_dim)
        self.band_scale = nn.Parameter(torch.zeros(4))

    @staticmethod
    def _depthwise_conv(channels: int) -> nn.Conv2d:
        return nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def _mix(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ll, lh, hl, hh = self.dwt(x)
        residual_bands = (
            self.band_scale[0] * self.ll_conv(ll),
            self.band_scale[1] * self.lh_conv(lh),
            self.band_scale[2] * self.hl_conv(hl),
            self.band_scale[3] * self.hh_conv(hh),
        )
        residual = self.idwt(*residual_bands)
        diagnostics = {
            "wavelet_ll_scale": self.band_scale[0].abs(),
            "wavelet_lh_scale": self.band_scale[1].abs(),
            "wavelet_hl_scale": self.band_scale[2].abs(),
            "wavelet_hh_scale": self.band_scale[3].abs(),
            "wavelet_ll_energy": residual_bands[0].square().mean(dim=(1, 2, 3)),
            "wavelet_lh_energy": residual_bands[1].square().mean(dim=(1, 2, 3)),
            "wavelet_hl_energy": residual_bands[2].square().mean(dim=(1, 2, 3)),
            "wavelet_hh_energy": residual_bands[3].square().mean(dim=(1, 2, 3)),
        }
        return residual, diagnostics


ADAPTER_TYPES = {
    "spatial": SpatialTokenAdapter,
    "global_filter": GlobalFilterTokenAdapter,
    "factorized_filter": FactorizedSpectralTokenAdapter,
    "wavelet": WaveletTokenAdapter,
}


class InternalAdapterCLIPClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai",
        freeze_clip: bool = True,
        adapter_type: str = "spatial",
        adapter_indices: list[int] | tuple[int, ...] = (3, 7, 11),
        bottleneck_dim: int = 128,
        dropout: float = 0.2,
        adapter_dropout: float = 0.1,
        layer_scale_init: float = 1.0e-4,
        radial_bands: int = 6,
        orientation_bins: int = 6,
    ) -> None:
        super().__init__()
        if not freeze_clip:
            raise ValueError("Phase 7 internal adapters require freeze_clip=True.")
        if adapter_type not in ADAPTER_TYPES:
            raise ValueError(f"Unsupported adapter_type: {adapter_type}")

        self.clip_model = open_clip.create_model(clip_model, pretrained=clip_pretrained)
        self.freeze_clip = True
        self.adapter_type = adapter_type
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        self._has_trainable_backbone = False
        self.clip_model.eval()

        visual = self.clip_model.visual
        blocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
        if blocks is None:
            raise ValueError("OpenCLIP visual tower must expose transformer.resblocks.")
        if not bool(getattr(visual.transformer, "batch_first", False)):
            raise ValueError("Phase 7 manual forward requires batch-first OpenCLIP visual tokens.")
        self.adapter_indices = self._validate_adapter_indices(adapter_indices, len(blocks))
        self._first_adapter_index = self.adapter_indices[0]

        embed_dim = getattr(visual, "width", None)
        if embed_dim is None:
            embed_dim = getattr(getattr(visual, "conv1", None), "out_channels", None)
        output_dim = getattr(visual, "output_dim", None)
        if embed_dim is None or output_dim is None:
            raise ValueError("Could not infer OpenCLIP visual token or output dimension.")
        self.embed_dim = int(embed_dim)
        self.output_dim = int(output_dim)

        # Build the shared head before variant-specific modules so seed-matched
        # candidates start from identical classifier parameters.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, num_classes),
        )

        adapters: dict[str, nn.Module] = {}
        for index in self.adapter_indices:
            kwargs: dict[str, Any] = {
                "embed_dim": self.embed_dim,
                "bottleneck_dim": bottleneck_dim,
                "adapter_dropout": adapter_dropout,
                "layer_scale_init": layer_scale_init,
            }
            if adapter_type == "factorized_filter":
                kwargs.update(radial_bands=radial_bands, orientation_bins=orientation_bins)
            adapters[str(index)] = ADAPTER_TYPES[adapter_type](**kwargs)
        self.adapters = nn.ModuleDict(adapters)

    @staticmethod
    def _validate_adapter_indices(
        adapter_indices: list[int] | tuple[int, ...],
        num_blocks: int,
    ) -> tuple[int, ...]:
        indices = tuple(int(index) for index in adapter_indices)
        if not indices:
            raise ValueError("adapter_indices must not be empty.")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("adapter_indices must be unique and sorted in ascending order.")
        if indices[0] < 0 or indices[-1] >= num_blocks:
            raise ValueError(f"adapter_indices {indices} are invalid for {num_blocks} blocks.")
        return indices

    @property
    def has_trainable_backbone(self) -> bool:
        return self._has_trainable_backbone

    def train(self, mode: bool = True) -> InternalAdapterCLIPClassifier:
        super().train(mode)
        self.clip_model.eval()
        return self

    @staticmethod
    def _aggregate_diagnostic(values: list[torch.Tensor], batch_size: int) -> torch.Tensor:
        normalized = []
        for value in values:
            if value.ndim == 0:
                normalized.append(value)
            elif value.shape[0] == batch_size:
                normalized.append(value)
            else:
                normalized.append(value.mean())
        if all(value.ndim == 0 for value in normalized):
            return torch.stack(normalized).mean()
        per_sample = [
            value if value.ndim > 0 and value.shape[0] == batch_size else value.expand(batch_size)
            for value in normalized
        ]
        return torch.stack(per_sample, dim=1).mean(dim=1)

    def _apply_adapter(
        self,
        index: int,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cls_token = tokens[:, :1, :]
        patch_tokens = tokens[:, 1:, :]
        if patch_tokens.shape[1] != PATCH_TOKEN_COUNT:
            raise RuntimeError(
                f"Expected {PATCH_TOKEN_COUNT} spatial tokens, found {patch_tokens.shape[1]}."
            )
        adapted, diagnostics = self.adapters[str(index)](patch_tokens, return_diagnostics=True)
        return torch.cat([cls_token, adapted], dim=1), diagnostics

    def _encode_with_adapters(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        visual = self.clip_model.visual
        blocks = visual.transformer.resblocks
        per_layer_diagnostics: list[dict[str, torch.Tensor]] = []

        with torch.no_grad():
            tokens = visual._embeds(images)
            for index in range(self._first_adapter_index + 1):
                tokens = _run_block(blocks[index], tokens)
        tokens = tokens.detach()
        tokens, diagnostics = self._apply_adapter(self._first_adapter_index, tokens)
        per_layer_diagnostics.append(diagnostics)

        for index in range(self._first_adapter_index + 1, len(blocks)):
            tokens = _run_block(blocks[index], tokens)
            if index in self.adapter_indices:
                tokens, diagnostics = self._apply_adapter(index, tokens)
                per_layer_diagnostics.append(diagnostics)

        pooled, _ = visual._pool(tokens)
        if visual.proj is not None:
            pooled = pooled @ visual.proj

        batch_size = images.shape[0]
        combined: dict[str, list[torch.Tensor]] = {}
        for diagnostics in per_layer_diagnostics:
            for key, value in diagnostics.items():
                combined.setdefault(key, []).append(value)
        aggregated = {
            key: self._aggregate_diagnostic(values, batch_size)
            for key, values in combined.items()
            if key != "layer_scale_mean"
        }
        aggregated["adapter_layer_scales"] = torch.stack(
            [diagnostics["layer_scale_mean"] for diagnostics in per_layer_diagnostics]
        )
        return pooled, aggregated

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled, diagnostics = self._encode_with_adapters(images)
        logits = self.classifier(pooled.float())
        return {
            "logits": logits,
            "adapter_residual_abs_mean": diagnostics.pop("residual_abs_mean"),
            "adapter_residual_norm": diagnostics.pop("residual_norm"),
            **diagnostics,
        }


def count_trainable_parameters(model: InternalAdapterCLIPClassifier) -> dict[str, int]:
    return {
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
        "frozen_clip_params": sum(parameter.numel() for parameter in model.clip_model.parameters()),
        "adapter_params": sum(parameter.numel() for parameter in model.adapters.parameters()),
        "classifier_params": sum(parameter.numel() for parameter in model.classifier.parameters()),
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
