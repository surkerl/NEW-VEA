import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.emotionroi import EmotionROIDataset, discover_emotionroi_splits
from src.models import (
    AffectSpectrumFiLMClassifier,
    CLIPFFTConcatClassifier,
    CLIPLinearClassifier,
    FrequencyOnlyClassifier,
)
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, set_nested
from src.utils.metrics import accuracy_from_logits
from src.utils.seed import build_generator, build_worker_init_fn, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EmotionROI CLIP baseline checkpoint.")
    parser.add_argument("--config", default="configs/roi_clip_baseline.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run_name")
    parser.add_argument("--device")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for attr, path in {
        "data_root": ("dataset", "root"),
        "batch_size": ("train", "batch_size"),
        "num_workers": ("train", "num_workers"),
        "seed": ("train", "seed"),
        "run_name": ("logging", "run_name"),
    }.items():
        value = getattr(args, attr)
        if value is not None:
            set_nested(config, path, value)
    return config


def build_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    input_size = int(dataset_cfg.get("input_size", 224))
    model_name = model_cfg.get("name", "clip_linear")

    if model_name == "clip_linear":
        return CLIPLinearClassifier(
            num_classes=num_classes,
            model_name=model_cfg.get("clip_model", "ViT-B-16"),
            pretrained=model_cfg.get("clip_pretrained", "openai"),
            freeze_clip=bool(model_cfg.get("freeze_clip", True)),
            train_last_n_blocks=int(model_cfg.get("train_last_n_blocks", 0)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if model_name == "frequency_only":
        return FrequencyOnlyClassifier(
            num_classes=num_classes,
            input_size=input_size,
            num_bands=int(model_cfg.get("num_bands", 6)),
            num_orientations=int(model_cfg.get("num_orientations", 6)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if model_name == "clip_fft_concat":
        return CLIPFFTConcatClassifier(
            num_classes=num_classes,
            input_size=input_size,
            model_name=model_cfg.get("clip_model", "ViT-B-16"),
            pretrained=model_cfg.get("clip_pretrained", "openai"),
            freeze_clip=bool(model_cfg.get("freeze_clip", True)),
            train_last_n_blocks=int(model_cfg.get("train_last_n_blocks", 0)),
            num_bands=int(model_cfg.get("num_bands", 6)),
            num_orientations=int(model_cfg.get("num_orientations", 6)),
            spectral_dim=int(model_cfg.get("spectral_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if model_name == "affectspectrum_film":
        return AffectSpectrumFiLMClassifier(
            num_classes=num_classes,
            input_size=input_size,
            model_name=model_cfg.get("clip_model", "ViT-B-16"),
            pretrained=model_cfg.get("clip_pretrained", "openai"),
            freeze_clip=bool(model_cfg.get("freeze_clip", True)),
            train_last_n_blocks=int(model_cfg.get("train_last_n_blocks", 0)),
            num_bands=int(model_cfg.get("num_bands", 6)),
            num_orientations=int(model_cfg.get("num_orientations", 6)),
            spectral_hidden_dim=int(model_cfg.get("spectral_hidden_dim", 256)),
            film_scale=float(model_cfg.get("film_scale", 0.1)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    raise ValueError(f"Unsupported model.name: {model_name}")


def extract_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["logits"]
    return output


def load_checkpoint_state(model: nn.Module, checkpoint: dict[str, Any]) -> None:
    checkpoint_format = checkpoint.get("checkpoint_format", "full_model")
    state_dict = checkpoint["model_state_dict"]

    if checkpoint_format == "head_only_frozen_clip":
        result = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(result.missing_keys)
        unexpected_keys = list(result.unexpected_keys)
        disallowed_missing = [key for key in missing_keys if not key.startswith("clip_model.")]
        if disallowed_missing:
            raise RuntimeError(
                "Head-only checkpoint is missing classifier parameters: "
                + ", ".join(disallowed_missing)
            )
        if unexpected_keys:
            raise RuntimeError(
                "Head-only checkpoint has unexpected parameters: "
                + ", ".join(unexpected_keys)
            )
        return

    if checkpoint_format == "full_model":
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load full-model checkpoint with strict=True: {exc}") from exc
        return

    raise RuntimeError(f"Unsupported checkpoint_format: {checkpoint_format}")


def main() -> int:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    checkpoint = load_checkpoint(args.ckpt, map_location="cpu")
    ckpt_config = checkpoint.get("config", {})
    if ckpt_config:
        config["model"] = ckpt_config.get("model", config["model"])

    dataset_cfg = config["dataset"]
    train_cfg = config["train"]
    model_cfg = config["model"]
    run_name = args.run_name or config["logging"]["run_name"]

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed, deterministic=bool(train_cfg.get("deterministic", True)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    discovery = discover_emotionroi_splits(dataset_cfg["root"])
    test_dataset = EmotionROIDataset(
        dataset_cfg["root"],
        split="test",
        input_size=int(dataset_cfg.get("input_size", 224)),
        discovery=discovery,
    )
    num_workers = int(train_cfg.get("num_workers", 4))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg.get("batch_size", 32)),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": build_worker_init_fn(seed) if num_workers > 0 else None,
        "generator": build_generator(seed),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class_raw = checkpoint["idx_to_class"]
    idx_to_class = {int(key): str(value) for key, value in idx_to_class_raw.items()}
    num_classes = len(class_to_idx)

    model = build_model(config, num_classes=num_classes).to(device)
    load_checkpoint_state(model, checkpoint)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_path: list[str] = []
    y_prob: list[list[float]] = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, dynamic_ncols=True, ascii=True):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = extract_logits(model(images))
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            correct, _ = accuracy_from_logits(logits, labels)
            total_correct += correct
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits.float(), dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_path.extend([str(path) for path in batch["path"]])
            y_prob.extend(probs.cpu().tolist())

    test_acc = total_correct / max(total_count, 1)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    labels = list(range(num_classes))
    target_names = [idx_to_class[index] for index in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    result_dir = Path("results") / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / "eval_report.txt"
    predictions_path = result_dir / "predictions.csv"
    text = (
        f"checkpoint: {args.ckpt}\n"
        f"test_loss: {total_loss / max(total_count, 1):.6f}\n"
        f"test_acc: {test_acc:.6f}\n"
        f"macro_f1: {macro_f1:.6f}\n\n"
        f"{report}"
    )
    print(text)
    report_path.write_text(text, encoding="utf-8")
    print(f"Saved eval report: {report_path}")

    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "label", "pred", "correct"] + [f"prob_{index}" for index in range(num_classes)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for path, label, pred, probs in zip(y_path, y_true, y_pred, y_prob):
            row = {
                "path": path,
                "label": label,
                "pred": pred,
                "correct": int(label == pred),
            }
            row.update({f"prob_{index}": f"{prob:.8f}" for index, prob in enumerate(probs)})
            writer.writerow(row)
    print(f"Saved predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
