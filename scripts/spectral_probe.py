from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import build_model, extract_logits, load_checkpoint_state
from src.datasets.emotionroi import EmotionROIDataset, discover_emotionroi_splits
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.frequency_filters import (
    amplitude_noise,
    downsample_upsample,
    fft_band_pass,
    fft_high_pass,
    fft_low_pass,
    gaussian_blur_tensor,
    phase_noise,
)
from src.utils.seed import build_generator, build_worker_init_fn, set_seed


PerturbFn = Callable[[torch.Tensor, int | None], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spectral perturbation probes on official EmotionROI test split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--probe_name", required=True)
    parser.add_argument("--output_dir", default="results/phase3b_spectral_probe")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def model_label(model_name: str) -> str:
    if model_name == "clip_linear":
        return "baseline"
    if model_name == "clip_fft_concat":
        return "concat"
    if model_name == "affectspectrum_film":
        return "film"
    return model_name


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def perturbations() -> list[tuple[str, PerturbFn]]:
    return [
        ("full", lambda x, seed: x),
        ("low_0.15", lambda x, seed: fft_low_pass(x, 0.15)),
        ("low_0.30", lambda x, seed: fft_low_pass(x, 0.30)),
        ("low_0.50", lambda x, seed: fft_low_pass(x, 0.50)),
        ("high_0.30", lambda x, seed: fft_high_pass(x, 0.30)),
        ("high_0.50", lambda x, seed: fft_high_pass(x, 0.50)),
        ("band_0.15_0.30", lambda x, seed: fft_band_pass(x, 0.15, 0.30)),
        ("band_0.30_0.50", lambda x, seed: fft_band_pass(x, 0.30, 0.50)),
        ("band_0.50_0.75", lambda x, seed: fft_band_pass(x, 0.50, 0.75)),
        ("blur_light", lambda x, seed: gaussian_blur_tensor(x, kernel_size=9, sigma=2.0)),
        ("blur_heavy", lambda x, seed: gaussian_blur_tensor(x, kernel_size=17, sigma=5.0)),
        ("downsample_x2", lambda x, seed: downsample_upsample(x, factor=2)),
        ("downsample_x4", lambda x, seed: downsample_upsample(x, factor=4)),
        ("amplitude_noise_0.05", lambda x, seed: amplitude_noise(x, noise_std=0.05, seed=seed)),
        ("amplitude_noise_0.10", lambda x, seed: amplitude_noise(x, noise_std=0.10, seed=seed)),
        ("phase_noise_0.05", lambda x, seed: phase_noise(x, noise_std=0.05, seed=seed)),
        ("phase_noise_0.10", lambda x, seed: phase_noise(x, noise_std=0.10, seed=seed)),
    ]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite tensor detected: {name}")


def build_loader(config: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, dict[int, str]]:
    dataset_cfg = config["dataset"]
    train_cfg = config["train"]
    discovery = discover_emotionroi_splits(dataset_cfg["root"])
    dataset = EmotionROIDataset(
        dataset_cfg["root"],
        split="test",
        input_size=int(dataset_cfg.get("input_size", 224)),
        max_samples=args.max_test_samples,
        discovery=discovery,
    )
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 4))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": build_worker_init_fn(seed) if num_workers > 0 else None,
        "generator": build_generator(seed),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, shuffle=False, **loader_kwargs), dict(discovery.idx_to_class)


def evaluate_perturbation(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    transform_name: str,
    transform_fn: PerturbFn,
    output_dir: Path,
    probe_name: str,
    run_name: str,
    model_name: str,
    seed: int,
    idx_to_class: dict[int, str],
    perturbation_index: int,
    collect_response: bool,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray] | None]:
    num_classes = len(idx_to_class)
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, object]] = []
    class_counts = {idx: 0 for idx in idx_to_class}
    class_correct = {idx: 0 for idx in idx_to_class}
    gate_sum = 0.0
    gate_count = 0

    response_sum: torch.Tensor | None = None
    response_total = 0
    class_response_sum: torch.Tensor | None = None
    class_response_count: torch.Tensor | None = None

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, leave=False, dynamic_ncols=True, ascii=True, desc=transform_name)):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            perturb_seed = seed + perturbation_index * 100000 + batch_index
            perturbed = transform_fn(images, perturb_seed)
            assert_finite_tensor(f"{transform_name}_images", perturbed)

            output = model(perturbed)
            logits = extract_logits(output)
            assert_finite_tensor(f"{transform_name}_logits", logits)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits.float(), dim=1)
            assert_finite_tensor(f"{transform_name}_probs", probs)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            correct_mask = preds.eq(labels)
            total_correct += int(correct_mask.sum().item())
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

            for class_idx in idx_to_class:
                mask = labels == class_idx
                count = int(mask.sum().item())
                class_counts[class_idx] += count
                if count:
                    class_correct[class_idx] += int(correct_mask[mask].sum().item())

            if collect_response and isinstance(output, dict) and "response_map" in output:
                response = output["response_map"].detach().float().cpu()
                if response_sum is None:
                    response_sum = torch.zeros_like(response[0])
                    class_response_sum = torch.zeros_like(response[0])
                    class_response_count = torch.zeros(response.size(1), dtype=torch.float32)
                response_sum += response.sum(dim=0)
                response_total += response.size(0)
                assert class_response_sum is not None and class_response_count is not None
                labels_cpu = labels.cpu()
                for class_idx in range(response.size(1)):
                    mask = labels_cpu == class_idx
                    if bool(mask.any()):
                        class_response_sum[class_idx] += response[mask, class_idx].sum(dim=0)
                        class_response_count[class_idx] += float(mask.sum().item())
            if isinstance(output, dict) and "gate" in output:
                gate_values = output["gate"].detach().float()
                gate_sum += float(gate_values.sum().item())
                gate_count += int(gate_values.numel())

            paths = [str(path) for path in batch["path"]]
            labels_cpu = labels.cpu().tolist()
            preds_cpu = preds.cpu().tolist()
            probs_cpu = probs.cpu().tolist()
            for path, label, pred, prob_values in zip(paths, labels_cpu, preds_cpu, probs_cpu):
                row = {"path": path, "label": label, "pred": pred, "correct": int(label == pred)}
                row.update({f"prob_{index}": f"{prob:.8f}" for index, prob in enumerate(prob_values)})
                prediction_rows.append(row)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metric_row = {
        "probe_name": probe_name,
        "run_name": run_name,
        "model": model_name,
        "seed": seed,
        "perturbation": transform_name,
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
        "macro_f1": macro_f1,
        "num_samples": total_count,
        "mean_gate": gate_sum / gate_count if gate_count > 0 else "",
    }
    per_class_rows = [
        {
            "probe_name": probe_name,
            "run_name": run_name,
            "model": model_name,
            "seed": seed,
            "perturbation": transform_name,
            "class_idx": class_idx,
            "class_name": idx_to_class[class_idx],
            "num_samples": class_counts[class_idx],
            "acc": class_correct[class_idx] / max(class_counts[class_idx], 1),
        }
        for class_idx in sorted(idx_to_class)
    ]
    prediction_fields = ["path", "label", "pred", "correct"] + [f"prob_{index}" for index in range(num_classes)]
    write_csv(output_dir / f"predictions_{transform_name}.csv", prediction_rows, prediction_fields)

    response_payload = None
    if collect_response and response_sum is not None and class_response_sum is not None and class_response_count is not None:
        mean_response = (response_sum / max(response_total, 1)).numpy()
        counts = class_response_count.view(-1, 1, 1).clamp_min(1.0)
        class_response = (class_response_sum / counts).numpy()
        response_payload = {
            "mean_response_map": mean_response,
            "class_response_maps": class_response,
        }
    return metric_row, per_class_rows, response_payload


def save_response_maps(output_dir: Path, idx_to_class: dict[int, str], payload: dict[str, np.ndarray]) -> None:
    mean_response = payload["mean_response_map"]
    class_response = payload["class_response_maps"]
    np.save(output_dir / "mean_response_map.npy", mean_response)
    np.save(output_dir / "class_response_maps.npy", class_response)
    for class_idx, class_name in idx_to_class.items():
        if class_idx >= class_response.shape[0]:
            continue
        plt.figure(figsize=(5, 4))
        plt.imshow(class_response[class_idx], cmap="magma", aspect="auto")
        plt.colorbar(label="response")
        plt.xlabel("orientation bin")
        plt.ylabel("radial band")
        plt.title(f"FiLM spectral response: {class_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"response_map_class_{safe_name(class_name)}.png", dpi=160)
        plt.close()


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.float().clamp_min(1.0e-12)
    return -(probs * probs.log()).sum(dim=1)


def run_evidence_accumulation(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    idx_to_class: dict[int, str],
    seed: int,
) -> None:
    stages: list[tuple[str, PerturbFn]] = [
        ("low_0.15", lambda x, probe_seed: fft_low_pass(x, 0.15)),
        ("low_0.30", lambda x, probe_seed: fft_low_pass(x, 0.30)),
        ("low_0.50", lambda x, probe_seed: fft_low_pass(x, 0.50)),
        ("full", lambda x, probe_seed: x),
    ]
    sample_rows: list[dict[str, object]] = []
    aggregate: dict[tuple[int, str], dict[str, float]] = {}

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, leave=False, dynamic_ncols=True, ascii=True, desc="evidence")):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            paths = [str(path) for path in batch["path"]]
            for stage_index, (stage_name, stage_fn) in enumerate(stages):
                perturbed = stage_fn(images, seed + 500000 + stage_index * 100000 + batch_index)
                output = model(perturbed)
                logits = extract_logits(output)
                probs = torch.softmax(logits.float(), dim=1)
                entropy = entropy_from_probs(probs)
                preds = probs.argmax(dim=1)
                if isinstance(output, dict) and "gate" in output:
                    gate_values = output["gate"].detach().float().view(-1).cpu().tolist()
                else:
                    gate_values = [None] * labels.size(0)
                gt_probs = probs.gather(1, labels.view(-1, 1)).squeeze(1)
                pred_probs = probs.gather(1, preds.view(-1, 1)).squeeze(1)
                labels_cpu = labels.cpu().tolist()
                preds_cpu = preds.cpu().tolist()
                gt_cpu = gt_probs.cpu().tolist()
                pred_cpu = pred_probs.cpu().tolist()
                entropy_cpu = entropy.cpu().tolist()
                for path, label, pred, gt_prob, pred_prob, entropy_value, gate_value in zip(
                    paths, labels_cpu, preds_cpu, gt_cpu, pred_cpu, entropy_cpu, gate_values
                ):
                    correct = int(label == pred)
                    sample_rows.append(
                        {
                            "path": path,
                            "label": label,
                            "class_name": idx_to_class[int(label)],
                            "stage": stage_name,
                            "gt_prob": gt_prob,
                            "pred_prob": pred_prob,
                            "pred": pred,
                            "correct": correct,
                            "entropy": entropy_value,
                            "gate": "" if gate_value is None else gate_value,
                        }
                    )
                    key = (int(label), stage_name)
                    stats = aggregate.setdefault(key, {"gt_prob": 0.0, "entropy": 0.0, "correct": 0.0, "count": 0.0})
                    stats["gt_prob"] += float(gt_prob)
                    stats["entropy"] += float(entropy_value)
                    stats["correct"] += float(correct)
                    stats["count"] += 1.0

    by_class_rows = []
    for class_idx in sorted(idx_to_class):
        for stage_name, _ in stages:
            stats = aggregate.get((class_idx, stage_name), {"gt_prob": 0.0, "entropy": 0.0, "correct": 0.0, "count": 0.0})
            count = max(stats["count"], 1.0)
            by_class_rows.append(
                {
                    "class_idx": class_idx,
                    "class_name": idx_to_class[class_idx],
                    "stage": stage_name,
                    "mean_gt_prob": stats["gt_prob"] / count,
                    "mean_entropy": stats["entropy"] / count,
                    "acc": stats["correct"] / count,
                }
            )

    write_csv(output_dir / "evidence_accumulation_samples.csv", sample_rows)
    write_csv(output_dir / "evidence_accumulation_by_class.csv", by_class_rows)

    stage_names = [stage for stage, _ in stages]
    x = list(range(len(stage_names)))
    plt.figure(figsize=(8, 5))
    for class_idx in sorted(idx_to_class):
        values = [
            float(row["mean_gt_prob"])
            for row in by_class_rows
            if int(row["class_idx"]) == class_idx
        ]
        plt.plot(x, values, marker="o", label=idx_to_class[class_idx])
    plt.xticks(x, stage_names)
    plt.ylabel("mean ground-truth probability")
    plt.xlabel("coarse-to-fine spectral stage")
    plt.title("Coarse-to-fine spectral evidence accumulation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "evidence_accumulation_by_class.png", dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.ckpt, map_location="cpu")
    ckpt_config = checkpoint.get("config", {})
    if ckpt_config:
        config["model"] = ckpt_config.get("model", config["model"])
    config["logging"]["run_name"] = args.run_name
    config["train"]["seed"] = int(args.seed)
    if args.batch_size is not None:
        config["train"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)

    set_seed(int(args.seed), deterministic=bool(config["train"].get("deterministic", True)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader, discovered_idx_to_class = build_loader(config, args, device)
    idx_to_class_raw = checkpoint.get("idx_to_class", discovered_idx_to_class)
    idx_to_class = {int(key): str(value) for key, value in idx_to_class_raw.items()}
    num_classes = len(idx_to_class)

    model = build_model(config, num_classes=num_classes).to(device)
    load_checkpoint_state(model, checkpoint)
    model.eval()

    out_dir = PROJECT_ROOT / args.output_dir / args.probe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    run_model = model_label(str(config["model"].get("name", "clip_linear")))

    metric_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    response_payload: dict[str, np.ndarray] | None = None
    for index, (name, fn) in enumerate(perturbations()):
        metric_row, class_rows, response = evaluate_perturbation(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            transform_name=name,
            transform_fn=fn,
            output_dir=out_dir,
            probe_name=args.probe_name,
            run_name=args.run_name,
            model_name=run_model,
            seed=int(args.seed),
            idx_to_class=idx_to_class,
            perturbation_index=index,
            collect_response=name == "full",
        )
        metric_rows.append(metric_row)
        per_class_rows.extend(class_rows)
        if response is not None:
            response_payload = response

    write_csv(out_dir / "spectral_probe_metrics.csv", metric_rows)
    write_csv(out_dir / "per_class_spectral_probe.csv", per_class_rows)
    if response_payload is not None:
        save_response_maps(out_dir, idx_to_class, response_payload)

    run_evidence_accumulation(
        model=model,
        loader=loader,
        device=device,
        output_dir=out_dir,
        idx_to_class=idx_to_class,
        seed=int(args.seed),
    )

    print(f"Saved spectral probe outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
