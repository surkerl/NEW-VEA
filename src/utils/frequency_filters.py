"""Tensor frequency perturbations for CLIP-normalized RGB images."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
EPS = 1.0e-6


def _clip_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return mean, std


def denormalize_clip(x: torch.Tensor) -> torch.Tensor:
    """Convert CLIP-normalized RGB tensors to clamped [0, 1] image tensors."""
    x = x.float()
    mean, std = _clip_stats(x)
    return (x * std + mean).clamp(0.0, 1.0)


def normalize_clip(x: torch.Tensor) -> torch.Tensor:
    """Convert [0, 1] RGB tensors back to CLIP-normalized tensors."""
    x = x.float().clamp(0.0, 1.0)
    mean, std = _clip_stats(x)
    return (x - mean) / std


def _radius_mask(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    low: float | None = None,
    high: float | None = None,
) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    radius = torch.sqrt(grid_x.square() + grid_y.square()) / (2.0**0.5)
    mask = torch.ones_like(radius, dtype=torch.bool)
    if low is not None:
        mask = mask & (radius >= float(low))
    if high is not None:
        mask = mask & (radius <= float(high))
    return mask.view(1, 1, height, width)


def _fft_filter(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x01 = denormalize_clip(x)
    spectrum = torch.fft.fft2(x01.float(), dim=(-2, -1))
    shifted = torch.fft.fftshift(spectrum, dim=(-2, -1))
    filtered = shifted * mask.to(dtype=shifted.dtype)
    restored = torch.fft.ifftshift(filtered, dim=(-2, -1))
    output = torch.fft.ifft2(restored, dim=(-2, -1)).real
    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return normalize_clip(output)


def fft_low_pass(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Keep frequencies with normalized radius <= cutoff."""
    _, _, height, width = x.shape
    cutoff = max(0.0, min(1.0, float(cutoff)))
    mask = _radius_mask(height, width, x.device, torch.float32, high=cutoff)
    return _fft_filter(x, mask)


def fft_high_pass(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Keep frequencies with normalized radius >= cutoff."""
    _, _, height, width = x.shape
    cutoff = max(0.0, min(1.0, float(cutoff)))
    mask = _radius_mask(height, width, x.device, torch.float32, low=cutoff)
    return _fft_filter(x, mask)


def fft_band_pass(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Keep frequencies with normalized radius between low and high."""
    _, _, height, width = x.shape
    low = max(0.0, min(1.0, float(low)))
    high = max(low, min(1.0, float(high)))
    mask = _radius_mask(height, width, x.device, torch.float32, low=low, high=high)
    return _fft_filter(x, mask)


def _randn_like(x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    if seed is None:
        return torch.randn_like(x)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def amplitude_noise(x: torch.Tensor, noise_std: float, seed: int | None = None) -> torch.Tensor:
    """Apply multiplicative Fourier amplitude noise while preserving phase."""
    x01 = denormalize_clip(x)
    spectrum = torch.fft.fft2(x01.float(), dim=(-2, -1))
    amplitude = torch.abs(spectrum)
    phase = torch.angle(spectrum)
    eps = _randn_like(amplitude, seed=seed)
    amplitude_new = amplitude * torch.exp(float(noise_std) * eps)
    output = torch.fft.ifft2(torch.polar(amplitude_new, phase), dim=(-2, -1)).real
    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return normalize_clip(output)


def phase_noise(x: torch.Tensor, noise_std: float, seed: int | None = None) -> torch.Tensor:
    """Apply additive Fourier phase noise while preserving amplitude."""
    x01 = denormalize_clip(x)
    spectrum = torch.fft.fft2(x01.float(), dim=(-2, -1))
    amplitude = torch.abs(spectrum)
    phase = torch.angle(spectrum)
    eps = _randn_like(phase, seed=seed)
    phase_new = phase + float(noise_std) * eps
    output = torch.fft.ifft2(torch.polar(amplitude, phase_new), dim=(-2, -1)).real
    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return normalize_clip(output)


def downsample_upsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample by factor and upsample back to the original spatial size."""
    x01 = denormalize_clip(x)
    factor = max(1, int(factor))
    height, width = x01.shape[-2:]
    small_h = max(1, height // factor)
    small_w = max(1, width // factor)
    small = F.interpolate(x01, size=(small_h, small_w), mode="bicubic", align_corners=False)
    output = F.interpolate(small, size=(height, width), mode="bicubic", align_corners=False)
    return normalize_clip(torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0))


def gaussian_blur_tensor(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to CLIP-normalized image tensors."""
    x01 = denormalize_clip(x)
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    output = TF.gaussian_blur(x01, kernel_size=[kernel_size, kernel_size], sigma=[float(sigma), float(sigma)])
    return normalize_clip(torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0))
