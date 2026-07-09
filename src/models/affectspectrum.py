import torch
from torch import nn

import open_clip

from .spectral import RadialOrientationSpectrum


def _infer_clip_feature_dim(clip_model: nn.Module) -> int:
    feature_dim = getattr(clip_model.visual, "output_dim", None)
    if feature_dim is None:
        feature_dim = getattr(clip_model, "embed_dim", None)
    if feature_dim is None:
        raise ValueError("Could not infer CLIP visual feature dimension.")
    return int(feature_dim)


class _FrozenClipMixin:
    def _init_clip(self, model_name: str, pretrained: str, freeze_clip: bool, train_last_n_blocks: int) -> int:
        self.clip_model = open_clip.create_model(model_name, pretrained=pretrained)
        self.freeze_clip = bool(freeze_clip)
        self.train_last_n_blocks = int(train_last_n_blocks)
        self._has_trainable_backbone = False
        self._configure_clip_trainability()
        return _infer_clip_feature_dim(self.clip_model)

    @property
    def has_trainable_backbone(self) -> bool:
        return self._has_trainable_backbone

    def backbone_parameters(self):
        return (param for param in self.clip_model.parameters() if param.requires_grad)

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if self.has_trainable_backbone:
            features = self.clip_model.encode_image(images)
        else:
            with torch.no_grad():
                features = self.clip_model.encode_image(images)
        return features.float()

    def _configure_clip_trainability(self) -> None:
        for param in self.clip_model.parameters():
            param.requires_grad = False

        if self.train_last_n_blocks > 0:
            self._unfreeze_last_visual_blocks(self.train_last_n_blocks)
        elif not self.freeze_clip:
            for param in self.clip_model.visual.parameters():
                param.requires_grad = True
        self._has_trainable_backbone = any(param.requires_grad for param in self.clip_model.parameters())

    def _unfreeze_last_visual_blocks(self, n_blocks: int) -> None:
        visual = self.clip_model.visual
        transformer = getattr(visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        if blocks is None:
            raise ValueError("train_last_n_blocks requires a CLIP visual transformer with resblocks.")

        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        for attr_name in ("ln_post", "proj"):
            if hasattr(visual, attr_name):
                self._set_trainable(getattr(visual, attr_name))

    @staticmethod
    def _set_trainable(obj) -> None:
        if isinstance(obj, nn.Parameter):
            obj.requires_grad = True
        elif isinstance(obj, nn.Module):
            for param in obj.parameters():
                param.requires_grad = True
        elif isinstance(obj, torch.Tensor) and hasattr(obj, "requires_grad"):
            obj.requires_grad = True


class FrequencyOnlyClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = 224,
        num_bands: int = 6,
        num_orientations: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.spectrum = RadialOrientationSpectrum(
            input_size=input_size,
            num_bands=num_bands,
            num_orientations=num_orientations,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.spectrum.output_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        spectral = self.spectrum(images)
        return self.classifier(spectral["spectral_vec"])


class SpectralPresentationEncoder(nn.Module):
    def __init__(
        self,
        spectral_input_dim: int,
        spectral_feature_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(spectral_input_dim),
            nn.Linear(spectral_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, spectral_feature_dim),
            nn.GELU(),
            nn.LayerNorm(spectral_feature_dim),
        )

    def forward(self, spectral_vec: torch.Tensor, spectral_map: torch.Tensor) -> torch.Tensor:
        del spectral_map
        return self.encoder(spectral_vec)


class CLIPFFTConcatClassifier(_FrozenClipMixin, nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = 224,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_clip: bool = True,
        train_last_n_blocks: int = 0,
        num_bands: int = 6,
        num_orientations: int = 6,
        spectral_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        clip_dim = self._init_clip(model_name, pretrained, freeze_clip, train_last_n_blocks)
        self.spectrum = RadialOrientationSpectrum(
            input_size=input_size,
            num_bands=num_bands,
            num_orientations=num_orientations,
        )
        self.spectral_mlp = nn.Sequential(
            nn.Linear(self.spectrum.output_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, spectral_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(clip_dim + spectral_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_feature = self._encode_image(images)
        spectral = self.spectrum(images)
        spectral_feature = self.spectral_mlp(spectral["spectral_vec"])
        fused = torch.cat([image_feature, spectral_feature], dim=-1)
        return self.classifier(fused)


class AffectSpectrumFiLMClassifier(_FrozenClipMixin, nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = 224,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_clip: bool = True,
        train_last_n_blocks: int = 0,
        num_bands: int = 6,
        num_orientations: int = 6,
        spectral_hidden_dim: int = 256,
        film_scale: float = 0.1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_bands = int(num_bands)
        self.num_orientations = int(num_orientations)
        self.film_scale = float(film_scale)
        clip_dim = self._init_clip(model_name, pretrained, freeze_clip, train_last_n_blocks)
        self.spectrum = RadialOrientationSpectrum(
            input_size=input_size,
            num_bands=num_bands,
            num_orientations=num_orientations,
        )
        self.spectral_mlp = nn.Sequential(
            nn.Linear(self.spectrum.output_dim, spectral_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(spectral_hidden_dim, spectral_hidden_dim),
            nn.GELU(),
        )
        self.gamma = nn.Linear(spectral_hidden_dim, clip_dim)
        self.beta = nn.Linear(spectral_hidden_dim, clip_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(clip_dim, num_classes),
        )
        self.response_head = nn.Linear(
            spectral_hidden_dim,
            self.num_classes * self.num_bands * self.num_orientations,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        image_feature = self._encode_image(images)
        spectral = self.spectrum(images)
        spectral_feature = self.spectral_mlp(spectral["spectral_vec"])
        gamma = torch.tanh(self.gamma(spectral_feature))
        beta = torch.tanh(self.beta(spectral_feature))
        modulated_feature = image_feature * (1.0 + self.film_scale * gamma) + self.film_scale * beta
        logits = self.classifier(modulated_feature)
        response_map = self.response_head(spectral_feature).view(
            images.size(0),
            self.num_classes,
            self.num_bands,
            self.num_orientations,
        )
        return {
            "logits": logits,
            "response_map": response_map,
            "spectral_map": spectral["spectral_map"],
        }


class AffectSpectrumGatedClassifier(_FrozenClipMixin, nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = 224,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_clip: bool = True,
        train_last_n_blocks: int = 0,
        num_bands: int = 6,
        num_orientations: int = 6,
        radial_spacing: str = "linear",
        spectral_feature_dim: int = 256,
        spectral_hidden_dim: int = 256,
        fusion_hidden_dim: int = 256,
        gate_hidden_dim: int = 256,
        dropout: float = 0.3,
        residual_scale: float = 0.25,
        response_logit_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_bands = int(num_bands)
        self.num_orientations = int(num_orientations)
        self.residual_scale = float(residual_scale)
        self.response_logit_scale = float(response_logit_scale)

        clip_dim = self._init_clip(model_name, pretrained, freeze_clip, train_last_n_blocks)
        self.spectrum = RadialOrientationSpectrum(
            input_size=input_size,
            num_bands=num_bands,
            num_orientations=num_orientations,
            radial_spacing=radial_spacing,
        )
        self.sem_norm = nn.LayerNorm(clip_dim)
        self.spectral_encoder = SpectralPresentationEncoder(
            spectral_input_dim=self.spectrum.output_dim,
            spectral_feature_dim=spectral_feature_dim,
            hidden_dim=spectral_hidden_dim,
            dropout=dropout,
        )
        self.spectral_to_sem = nn.Linear(spectral_feature_dim, clip_dim)
        self.spectral_sem_norm = nn.LayerNorm(clip_dim)
        self.response_head = nn.Linear(
            spectral_feature_dim,
            self.num_classes * self.num_bands * self.num_orientations,
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * clip_dim + self.num_classes, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        fusion_input_dim = clip_dim + spectral_feature_dim + clip_dim + self.num_classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_dim, self.num_classes),
        )
        self._has_trainable_backbone = any(param.requires_grad for param in self.clip_model.parameters())

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        z_sem = self._encode_image(images)
        sem_norm = self.sem_norm(z_sem)

        spectral = self.spectrum(images)
        spectral_map = spectral["spectral_map"]
        z_spec = self.spectral_encoder(spectral["spectral_vec"], spectral_map)
        z_spec_sem = self.spectral_sem_norm(self.spectral_to_sem(z_spec))

        batch_size = images.size(0)
        response_map = self.response_head(z_spec).view(
            batch_size,
            self.num_classes,
            self.num_bands,
            self.num_orientations,
        )
        energy = spectral_map.reshape(batch_size, self.num_bands * self.num_orientations)
        energy = energy / (energy.sum(dim=1, keepdim=True) + 1.0e-6)
        response_flat = response_map.reshape(batch_size, self.num_classes, self.num_bands * self.num_orientations)
        response_logits = (response_flat * energy.unsqueeze(1)).sum(dim=-1)

        gate_input = torch.cat(
            [
                sem_norm,
                z_spec_sem,
                sem_norm * z_spec_sem,
                response_logits,
            ],
            dim=-1,
        )
        gate = self.gate_mlp(gate_input)
        z_gated = sem_norm + self.residual_scale * gate * z_spec_sem
        fusion_input = torch.cat(
            [
                sem_norm,
                z_spec,
                z_gated,
                response_logits,
            ],
            dim=-1,
        )
        logits_main = self.classifier(fusion_input)
        logits = logits_main + self.response_logit_scale * response_logits

        return {
            "logits": logits,
            "logits_main": logits_main,
            "response_logits": response_logits,
            "response_map": response_map,
            "spectral_map": spectral_map,
            "gate": gate,
        }
