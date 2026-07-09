import argparse
import csv
import time
from pathlib import Path
from typing import Any

import torch
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
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, set_nested
from src.utils.logger import setup_logger
from src.utils.metrics import AverageMeter, accuracy_from_logits
from src.utils.seed import build_generator, build_worker_init_fn, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP visual encoder baseline on EmotionROI.")
    parser.add_argument("--config", default="configs/roi_clip_baseline.yaml")
    parser.add_argument("--data_root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--head_lr", type=float)
    parser.add_argument("--backbone_lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--run_name")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides = {
        "data_root": ("dataset", "root"),
        "epochs": ("train", "epochs"),
        "batch_size": ("train", "batch_size"),
        "head_lr": ("train", "head_lr"),
        "backbone_lr": ("train", "backbone_lr"),
        "patience": ("train", "patience"),
        "seed": ("train", "seed"),
        "num_workers": ("train", "num_workers"),
        "run_name": ("logging", "run_name"),
    }
    for attr, path in overrides.items():
        value = getattr(args, attr)
        if value is not None:
            set_nested(config, path, value)
    if args.lr is not None:
        set_nested(config, ("train", "head_lr"), args.lr)
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


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    model_name = config["model"].get("name", "clip_linear")
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-4))
    head_lr = float(train_cfg.get("head_lr", 1.0e-3))
    spectral_lr = float(train_cfg.get("spectral_lr", head_lr))
    backbone_lr = float(train_cfg.get("backbone_lr", 1.0e-5))

    if model_name == "frequency_only":
        lr = float(train_cfg.get("lr", 1.0e-3))
        params = [param for param in model.parameters() if param.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    backbone_params = [
        param for name, param in model.named_parameters() if name.startswith("clip_model.") and param.requires_grad
    ]
    non_clip_params = [
        param for name, param in model.named_parameters() if not name.startswith("clip_model.") and param.requires_grad
    ]

    param_groups: list[dict[str, Any]] = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    if non_clip_params:
        param_groups.append({"params": non_clip_params, "lr": spectral_lr, "name": "non_clip"})
    if not param_groups:
        raise ValueError("No trainable parameters found.")
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return max(float(group["lr"]) for group in optimizer.param_groups)


def build_model_state_for_checkpoint(model: nn.Module) -> tuple[dict[str, torch.Tensor], str]:
    if hasattr(model, "has_trainable_backbone") and not model.has_trainable_backbone:
        state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if not key.startswith("clip_model.")
        }
        return state, "head_only_frozen_clip"

    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return state, "full_model"


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, float, bool]:
    is_train = optimizer is not None
    model.train(is_train)
    if (
        is_train
        and hasattr(model, "has_trainable_backbone")
        and not model.has_trainable_backbone
        and hasattr(model, "clip_model")
    ):
        model.clip_model.eval()

    loss_meter = AverageMeter()
    correct = 0
    total = 0
    optimizer_step_done = False
    progress = tqdm(loader, leave=False, dynamic_ncols=True, ascii=True)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = extract_logits(model(images))
                loss = criterion(logits, labels)

            if is_train:
                assert scaler is not None
                if use_amp:
                    scale_before = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    optimizer_step_done = optimizer_step_done or scale_after >= scale_before
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer_step_done = True

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        batch_correct, batch_total = accuracy_from_logits(logits.detach(), labels)
        correct += batch_correct
        total += batch_total

    return loss_meter.avg, correct / max(total, 1), optimizer_step_done


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_acc: float,
    discovery,
    config: dict[str, Any],
) -> dict[str, Any]:
    model_state_dict, checkpoint_format = build_model_state_for_checkpoint(model)
    return {
        "model_state_dict": model_state_dict,
        "checkpoint_format": checkpoint_format,
        "clip_model": config["model"].get("clip_model"),
        "clip_pretrained": config["model"].get("clip_pretrained"),
        "freeze_clip": bool(config["model"].get("freeze_clip", True)),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "class_to_idx": discovery.class_to_idx,
        "idx_to_class": discovery.idx_to_class,
        "config": config,
    }


def main() -> int:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    run_name = config["logging"]["run_name"]

    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    log_path = Path("logs") / f"{run_name}.log"
    logger = setup_logger(run_name, log_path)

    seed = int(train_cfg.get("seed", 42))
    set_seed(
        seed,
        deterministic=bool(train_cfg.get("deterministic", True)),
        use_deterministic_algorithms=bool(train_cfg.get("use_deterministic_algorithms", False)),
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    logger.info(f"Run: {run_name}")
    logger.info(f"Device: {device} | AMP: {use_amp}")
    logger.info("Protocol: official train/test only; no validation split is used.")

    discovery = discover_emotionroi_splits(dataset_cfg["root"])
    train_dataset = EmotionROIDataset(
        dataset_cfg["root"],
        split="train",
        input_size=int(dataset_cfg.get("input_size", 224)),
        max_samples=args.max_train_samples,
        discovery=discovery,
    )
    test_dataset = EmotionROIDataset(
        dataset_cfg["root"],
        split="test",
        input_size=int(dataset_cfg.get("input_size", 224)),
        max_samples=args.max_test_samples,
        discovery=discovery,
    )
    logger.info(
        f"Split: {discovery.split_type} | train={len(train_dataset)} test={len(test_dataset)} classes={len(discovery.class_to_idx)}"
    )

    generator = build_generator(seed)
    worker_init_fn = build_worker_init_fn(seed)
    num_workers = int(train_cfg.get("num_workers", 4))
    batch_size = int(train_cfg.get("batch_size", 32))
    pin_memory = device.type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn if num_workers > 0 else None,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    model = build_model(config, num_classes=len(discovery.class_to_idx)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, config)
    epochs = int(train_cfg.get("epochs", 100))
    min_lr = float(train_cfg.get("min_lr", 1.0e-6))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    patience = int(train_cfg.get("patience", 25))

    result_dir = Path("results") / run_name
    ckpt_dir = Path("checkpoints") / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / "metrics.csv"
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
                "lr",
                "elapsed_sec",
                "compute_elapsed_sec",
                "checkpoint_elapsed_sec",
                "epoch_elapsed_sec",
                "patience_counter",
            ]
        )

    best_acc = float("-inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        compute_start_time = time.time()
        train_loss, train_acc, optimizer_step_done = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        test_loss, test_acc, _ = run_epoch(model, test_loader, criterion, device, use_amp=False)
        compute_elapsed = time.time() - compute_start_time
        lr_value = current_lr(optimizer)

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if optimizer_step_done:
            scheduler.step()

        checkpoint_start_time = time.time()
        payload = checkpoint_payload(model, optimizer, scheduler, epoch, best_acc, discovery, config)
        save_checkpoint(last_path, payload)
        if is_best:
            save_checkpoint(best_path, payload)
        checkpoint_elapsed = time.time() - checkpoint_start_time
        epoch_elapsed = time.time() - epoch_start_time

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{test_loss:.6f}",
                    f"{test_acc:.6f}",
                    f"{lr_value:.8e}",
                    f"{epoch_elapsed:.3f}",
                    f"{compute_elapsed:.3f}",
                    f"{checkpoint_elapsed:.3f}",
                    f"{epoch_elapsed:.3f}",
                    patience_counter,
                ]
            )

        line = (
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"lr={lr_value:.2e} | patience={patience_counter}/{patience}"
        )
        if is_best:
            line += " (new best)"
        logger.info(line)

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    logger.info(f"best_acc={best_acc:.4f}")
    logger.info(f"best_epoch={best_epoch}")
    logger.info(f"best_checkpoint={best_path}")
    logger.info(f"log_path={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
