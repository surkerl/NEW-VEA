import torch
from torch import nn

import open_clip


class CLIPLinearClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_clip: bool = True,
        train_last_n_blocks: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.clip_model = open_clip.create_model(model_name, pretrained=pretrained)
        self.freeze_clip = bool(freeze_clip)
        self.train_last_n_blocks = int(train_last_n_blocks)
        self._has_trainable_backbone = False

        feature_dim = getattr(self.clip_model.visual, "output_dim", None)
        if feature_dim is None:
            feature_dim = getattr(self.clip_model, "embed_dim", None)
        if feature_dim is None:
            raise ValueError("Could not infer CLIP visual feature dimension.")

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(int(feature_dim), num_classes),
        )
        self._configure_trainable_parameters()

    @property
    def has_trainable_backbone(self) -> bool:
        return self._has_trainable_backbone

    def backbone_parameters(self):
        return (param for param in self.clip_model.parameters() if param.requires_grad)

    def head_parameters(self):
        return self.head.parameters()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.has_trainable_backbone:
            features = self.clip_model.encode_image(images)
        else:
            with torch.no_grad():
                features = self.clip_model.encode_image(images)
        features = features.float()
        return self.head(features)

    def _configure_trainable_parameters(self) -> None:
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
