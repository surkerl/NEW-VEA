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
