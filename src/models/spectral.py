import math

import torch
from torch import nn


class RadialOrientationSpectrum(nn.Module):
    def __init__(
        self,
        input_size: int = 224,
        num_bands: int = 6,
        num_orientations: int = 6,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.num_bands = int(num_bands)
        self.num_orientations = int(num_orientations)
        self.eps = float(eps)
        self.extra_feature_dim = 9
        self.output_dim = self.num_bands * self.num_orientations + self.extra_feature_dim

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean, persistent=False)
        self.register_buffer("clip_std", std, persistent=False)

        bin_masks, radial_masks, orientation_masks, radial_centers = self._build_masks()
        self.register_buffer("bin_masks", bin_masks, persistent=False)
        self.register_buffer("bin_counts", bin_masks.sum(dim=(-1, -2)).clamp_min(1.0), persistent=False)
        self.register_buffer("radial_masks", radial_masks, persistent=False)
        self.register_buffer("radial_counts", radial_masks.sum(dim=(-1, -2)).clamp_min(1.0), persistent=False)
        self.register_buffer("orientation_masks", orientation_masks, persistent=False)
        self.register_buffer("orientation_counts", orientation_masks.sum(dim=(-1, -2)).clamp_min(1.0), persistent=False)

        log_freq = torch.log(radial_centers.clamp_min(self.eps))
        log_freq_centered = log_freq - log_freq.mean()
        self.register_buffer("log_freq_centered", log_freq_centered, persistent=False)
        self.register_buffer(
            "log_freq_var",
            (log_freq_centered.pow(2).sum()).clamp_min(self.eps),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.float()
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f"Expected input size {self.input_size}x{self.input_size}, got {tuple(x.shape[-2:])}.")

        image = (x * self.clip_std + self.clip_mean).clamp(0.0, 1.0)
        gray = 0.2989 * image[:, 0] + 0.5870 * image[:, 1] + 0.1140 * image[:, 2]
        fft = torch.fft.fft2(gray)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        amplitude = torch.log1p(torch.abs(fft))

        spectral_sum = (amplitude[:, None, None] * self.bin_masks[None]).sum(dim=(-1, -2))
        spectral_map = spectral_sum / self.bin_counts[None]

        radial_sum = (amplitude[:, None] * self.radial_masks[None]).sum(dim=(-1, -2))
        radial_energy = radial_sum / self.radial_counts[None]
        orientation_sum = (amplitude[:, None] * self.orientation_masks[None]).sum(dim=(-1, -2))
        orientation_energy = orientation_sum / self.orientation_counts[None]

        flat_energy = spectral_map.flatten(1)
        total_energy = amplitude.mean(dim=(-1, -2), keepdim=False).unsqueeze(1)
        spectral_entropy = self._entropy(flat_energy)

        low_energy, mid_energy, high_energy = self._split_radial_energy(radial_energy)
        low_mid_ratio = low_energy / (mid_energy + self.eps)
        high_mid_ratio = high_energy / (mid_energy + self.eps)
        spectral_slope = self._spectral_slope(radial_energy)
        orientation_entropy = self._entropy(orientation_energy)

        extra_features = torch.cat(
            [
                total_energy,
                spectral_entropy,
                low_energy,
                mid_energy,
                high_energy,
                low_mid_ratio,
                high_mid_ratio,
                spectral_slope,
                orientation_entropy,
            ],
            dim=1,
        )
        spectral_vec = torch.cat([flat_energy, extra_features], dim=1)
        return {"spectral_map": spectral_map, "spectral_vec": spectral_vec}

    def _build_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        size = self.input_size
        coords = torch.arange(size, dtype=torch.float32) - (size // 2)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        radius = torch.sqrt(xx.pow(2) + yy.pow(2))
        valid = radius > 0

        max_radius = radius.max().clamp_min(1.0)
        radial_edges = torch.linspace(0.0, float(max_radius), self.num_bands + 1)
        radial_masks = []
        for band_idx in range(self.num_bands):
            lower = radial_edges[band_idx]
            upper = radial_edges[band_idx + 1]
            if band_idx == self.num_bands - 1:
                mask = (radius > lower) & (radius <= upper) & valid
            else:
                mask = (radius > lower) & (radius <= upper) & valid
            radial_masks.append(mask.float())
        radial_masks_t = torch.stack(radial_masks, dim=0)

        theta = torch.remainder(torch.atan2(yy, xx), math.pi)
        orientation_edges = torch.linspace(0.0, math.pi, self.num_orientations + 1)
        orientation_masks = []
        for orient_idx in range(self.num_orientations):
            lower = orientation_edges[orient_idx]
            upper = orientation_edges[orient_idx + 1]
            if orient_idx == self.num_orientations - 1:
                mask = (theta >= lower) & (theta <= upper) & valid
            else:
                mask = (theta >= lower) & (theta < upper) & valid
            orientation_masks.append(mask.float())
        orientation_masks_t = torch.stack(orientation_masks, dim=0)

        bin_masks = radial_masks_t[:, None] * orientation_masks_t[None]
        radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:]) / max_radius
        return bin_masks.float(), radial_masks_t.float(), orientation_masks_t.float(), radial_centers.float()

    def _entropy(self, energy: torch.Tensor) -> torch.Tensor:
        probs = energy.clamp_min(0.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(self.eps)
        entropy = -(probs * torch.log(probs.clamp_min(self.eps))).sum(dim=1, keepdim=True)
        max_entropy = math.log(max(energy.shape[1], 2))
        return entropy / max_entropy

    def _split_radial_energy(self, radial_energy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunks = torch.tensor_split(radial_energy, 3, dim=1)
        low = chunks[0].mean(dim=1, keepdim=True)
        mid = chunks[1].mean(dim=1, keepdim=True)
        high = chunks[2].mean(dim=1, keepdim=True)
        return low, mid, high

    def _spectral_slope(self, radial_energy: torch.Tensor) -> torch.Tensor:
        y = torch.log(radial_energy.clamp_min(self.eps))
        y_centered = y - y.mean(dim=1, keepdim=True)
        slope = (y_centered * self.log_freq_centered[None]).sum(dim=1, keepdim=True) / self.log_freq_var
        return slope
