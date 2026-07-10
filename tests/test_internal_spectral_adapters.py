from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import open_clip

from src.models.internal_spectral_adapters import (
    FactorizedSpectralTokenAdapter,
    GlobalFilterTokenAdapter,
    HaarDWT2D,
    HaarIDWT2D,
    InternalAdapterCLIPClassifier,
    SpatialTokenAdapter,
    WaveletTokenAdapter,
    _run_block,
)


class InternalSpectralAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.clip_model = open_clip.create_model("ViT-B-16", pretrained="openai").to(cls.device).eval()

    def test_manual_clip_forward_matches_official(self) -> None:
        torch.manual_seed(7)
        images = torch.randn(2, 3, 224, 224, device=self.device)
        visual = self.clip_model.visual
        with torch.no_grad():
            official = self.clip_model.encode_image(images, normalize=False)
            tokens = visual._embeds(images)
            for block in visual.transformer.resblocks:
                tokens = _run_block(block, tokens)
            manual, _ = visual._pool(tokens)
            if visual.proj is not None:
                manual = manual @ visual.proj
        self.assertLessEqual((official - manual).abs().max().item(), 1.0e-5)

    def test_adapter_indices_are_validated(self) -> None:
        self.assertEqual(
            InternalAdapterCLIPClassifier._validate_adapter_indices([3, 7, 11], 12),
            (3, 7, 11),
        )
        with self.assertRaises(ValueError):
            InternalAdapterCLIPClassifier._validate_adapter_indices([3, 12], 12)
        with self.assertRaises(ValueError):
            InternalAdapterCLIPClassifier._validate_adapter_indices([7, 3], 12)

    def test_openclip_has_196_patch_tokens(self) -> None:
        images = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            tokens = self.clip_model.visual._embeds(images)
        self.assertEqual(tuple(tokens.shape[1:]), (197, 768))
        self.assertEqual(tokens[:, 1:].shape[1], 196)

    def test_adapter_shapes_finite_and_small_at_initialization(self) -> None:
        torch.manual_seed(11)
        patch_tokens = torch.randn(2, 196, 64)
        adapters = [
            SpatialTokenAdapter(64, 16),
            GlobalFilterTokenAdapter(64, 16),
            FactorizedSpectralTokenAdapter(64, 16, radial_bands=6, orientation_bins=6),
            WaveletTokenAdapter(64, 16),
        ]
        for adapter in adapters:
            adapted, diagnostics = adapter(patch_tokens, return_diagnostics=True)
            self.assertEqual(adapted.shape, patch_tokens.shape)
            self.assertTrue(torch.isfinite(adapted).all().item())
            self.assertLess((adapted - patch_tokens).abs().mean().item(), 1.0e-2)
            self.assertAlmostEqual(adapter.gamma.abs().mean().item(), 1.0e-4, places=7)
            for value in diagnostics.values():
                self.assertTrue(torch.isfinite(value).all().item())

    def test_factorized_basis_is_partition_of_unity(self) -> None:
        adapter = FactorizedSpectralTokenAdapter(64, 16, radial_bands=6, orientation_bins=6)
        basis_sum = adapter.spectral_basis.sum(dim=(0, 1))
        self.assertTrue(torch.allclose(basis_sum, torch.ones_like(basis_sum), atol=1.0e-6))

    def test_seed_matched_classifiers_have_identical_initialization(self) -> None:
        reference = None
        for adapter_type in ("spatial", "global_filter", "factorized_filter", "wavelet"):
            torch.manual_seed(101)
            with patch(
                "src.models.internal_spectral_adapters.open_clip.create_model",
                return_value=self.clip_model,
            ):
                model = InternalAdapterCLIPClassifier(
                    num_classes=6,
                    adapter_type=adapter_type,
                    bottleneck_dim=128,
                )
            state = {key: value.detach().clone() for key, value in model.classifier.state_dict().items()}
            if reference is None:
                reference = state
            else:
                for key in reference:
                    self.assertTrue(torch.equal(reference[key], state[key]))

    def test_haar_dwt_idwt_reconstruction(self) -> None:
        torch.manual_seed(13)
        x = torch.randn(2, 8, 14, 14)
        reconstructed = HaarIDWT2D()(*HaarDWT2D()(x))
        max_abs_error = (x - reconstructed).abs().max().item()
        print(f"dwt_idwt_max_abs_error={max_abs_error:.10e}")
        self.assertLessEqual(max_abs_error, 1.0e-5)

    def test_internal_model_stays_close_to_unadapted_path(self) -> None:
        with patch(
            "src.models.internal_spectral_adapters.open_clip.create_model",
            return_value=self.clip_model,
        ):
            model = InternalAdapterCLIPClassifier(
                num_classes=6,
                adapter_type="spatial",
                adapter_indices=[3, 7, 11],
                bottleneck_dim=128,
                layer_scale_init=1.0e-4,
            ).to(self.device)
        model.train()
        self.assertFalse(model.clip_model.training)
        images = torch.randn(2, 3, 224, 224, device=self.device)
        with torch.no_grad():
            unadapted = model.clip_model.encode_image(images, normalize=False)
            adapted, diagnostics = model._encode_with_adapters(images)
            output = model(images)
        self.assertLess((adapted - unadapted).abs().mean().item(), 1.0e-2)
        self.assertAlmostEqual(
            diagnostics["adapter_layer_scales"].mean().item(),
            1.0e-4,
            places=7,
        )
        self.assertEqual(tuple(output["logits"].shape), (2, 6))
        for value in output.values():
            self.assertTrue(torch.isfinite(value).all().item())

    def test_gradient_reaches_first_adapter_but_not_clip(self) -> None:
        with patch(
            "src.models.internal_spectral_adapters.open_clip.create_model",
            return_value=self.clip_model,
        ):
            model = InternalAdapterCLIPClassifier(
                num_classes=6,
                adapter_type="global_filter",
                adapter_indices=[3, 7, 11],
                bottleneck_dim=32,
            ).to(self.device)
        model.train()
        model.zero_grad(set_to_none=True)
        images = torch.randn(1, 3, 224, 224, device=self.device)
        model(images)["logits"].sum().backward()
        first_gamma_grad = model.adapters["3"].gamma.grad
        self.assertIsNotNone(first_gamma_grad)
        self.assertTrue(torch.isfinite(first_gamma_grad).all().item())
        self.assertGreater(first_gamma_grad.abs().sum().item(), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.clip_model.parameters()))


if __name__ == "__main__":
    unittest.main()
