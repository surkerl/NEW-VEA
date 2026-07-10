from __future__ import annotations

from pathlib import Path

import torch

import open_clip


OUTPUT_PATH = Path("results/phase7_openclip_visual_inspection.txt")


def run_block(block: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    try:
        return block(tokens, attn_mask=None)
    except TypeError:
        return block(tokens)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = open_clip.create_model("ViT-B-16", pretrained="openai").to(device)
    model.eval()
    visual = model.visual

    width_attr = getattr(visual, "width", None)
    inferred_width = getattr(getattr(visual, "conv1", None), "out_channels", None)
    blocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
    required_attrs = ("_embeds", "transformer", "_pool", "proj")

    if blocks is None:
        raise RuntimeError("OpenCLIP visual transformer does not expose transformer.resblocks.")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    x = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        official = model.encode_image(x, normalize=False)
        tokens = visual._embeds(x)
        embed_shape = tuple(tokens.shape)
        for block in blocks:
            tokens = run_block(block, tokens)
        pooled, _ = visual._pool(tokens)
        if visual.proj is not None:
            pooled = pooled @ visual.proj

    diff = (official - pooled).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    patch_tokens = embed_shape[1] - 1
    has_expected_tokens = embed_shape[1] == 197 and patch_tokens == 196
    parity_passed = max_abs_diff <= 1.0e-5

    version = getattr(open_clip, "__version__", "unknown")
    lines = [
        f"open_clip_version: {version}",
        f"device: {device}",
        f"visual_class: {type(visual).__name__}",
        f"visual.width: {width_attr if width_attr is not None else '<missing>'}",
        f"visual.width_inferred: {inferred_width}",
        f"visual.patch_size: {getattr(visual, 'patch_size', '<missing>')}",
        f"visual.grid_size: {getattr(visual, 'grid_size', '<missing>')}",
        f"visual.output_dim: {getattr(visual, 'output_dim', '<missing>')}",
    ]
    lines.extend(
        f"visual.has_attr.{name}: {hasattr(visual, name)}"
        for name in required_attrs
    )
    lines.extend(
        [
            f"visual.has_transformer.resblocks: {blocks is not None}",
            f"transformer.batch_first: {getattr(visual.transformer, 'batch_first', '<missing>')}",
            f"resblocks_count: {len(blocks)}",
            f"block_0_class: {type(blocks[0]).__name__}",
            f"block_1_class: {type(blocks[1]).__name__}",
            f"block_2_class: {type(blocks[2]).__name__}",
            f"block_last_class: {type(blocks[-1]).__name__}",
            f"embeds_shape: {embed_shape}",
            f"token_count: {embed_shape[1]}",
            f"patch_token_count: {patch_tokens}",
            f"is_1_cls_plus_196_spatial: {has_expected_tokens}",
            f"encode_image_shape: {tuple(official.shape)}",
            f"manual_output_shape: {tuple(pooled.shape)}",
            f"max_abs_diff: {max_abs_diff:.10e}",
            f"mean_abs_diff: {mean_abs_diff:.10e}",
            f"manual_forward_parity_passed: {parity_passed}",
        ]
    )

    text = "\n".join(lines) + "\n"
    print(text, end="")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"Saved inspection: {OUTPUT_PATH}")

    missing = [name for name in required_attrs if not hasattr(visual, name)]
    if missing:
        raise RuntimeError(f"OpenCLIP visual tower is missing required attributes: {missing}")
    if len(blocks) != 12:
        raise RuntimeError(f"Expected 12 visual transformer blocks, found {len(blocks)}.")
    if not has_expected_tokens:
        raise RuntimeError(f"Expected 1 CLS + 196 spatial tokens, found shape {embed_shape}.")
    if not parity_passed:
        raise RuntimeError(
            "Manual OpenCLIP visual forward parity failed: "
            f"max_abs_diff={max_abs_diff:.10e}, mean_abs_diff={mean_abs_diff:.10e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
