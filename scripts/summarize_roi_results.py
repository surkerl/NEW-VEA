import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch


RUNS = [
    ("roi_clip_baseline_seed42_speedfix", "clip_linear"),
    ("roi_frequency_only_seed42", "frequency_only"),
    ("roi_clip_fft_concat_seed42", "clip_fft_concat"),
    ("roi_affectspectrum_film_seed42", "affectspectrum_film"),
]
BASELINE_RUN = "roi_clip_baseline_seed42_speedfix"


def read_metrics(run_name: str) -> list[dict[str, str]]:
    metrics_path = PROJECT_ROOT / "results" / run_name / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_macro_f1(run_name: str) -> float:
    report_path = PROJECT_ROOT / "results" / run_name / "eval_report.txt"
    text = report_path.read_text(encoding="utf-8")
    match = re.search(r"^macro_f1:\s*([0-9.]+)", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"macro_f1 not found in {report_path}")
    return float(match.group(1))


def early_stopped(run_name: str) -> bool:
    log_path = PROJECT_ROOT / "logs" / f"{run_name}.log"
    return "Early stopping triggered" in log_path.read_text(encoding="utf-8")


def summarize_run(run_name: str, model_name: str) -> dict[str, object]:
    rows = read_metrics(run_name)
    best_row = max(rows, key=lambda row: float(row["test_acc"]))
    final_row = rows[-1]
    ckpt_path = PROJECT_ROOT / "checkpoints" / run_name / "best.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})
    seed = config.get("train", {}).get("seed", 42)
    mean_epoch_time = sum(float(row.get("epoch_elapsed_sec", row.get("elapsed_sec", 0.0))) for row in rows) / len(rows)
    mean_checkpoint_time = sum(float(row.get("checkpoint_elapsed_sec", 0.0)) for row in rows) / len(rows)

    return {
        "run_name": run_name,
        "model": config.get("model", {}).get("name", model_name),
        "seed": seed,
        "best_epoch": int(best_row["epoch"]),
        "best_acc": float(best_row["test_acc"]),
        "final_epoch": int(final_row["epoch"]),
        "final_acc": float(final_row["test_acc"]),
        "macro_f1": read_macro_f1(run_name),
        "early_stopped": early_stopped(run_name),
        "best_ckpt": str(ckpt_path),
        "log_path": str(PROJECT_ROOT / "logs" / f"{run_name}.log"),
        "checkpoint_size_bytes": ckpt_path.stat().st_size,
        "mean_epoch_time_sec": mean_epoch_time,
        "mean_checkpoint_time_sec": mean_checkpoint_time,
    }


def write_summary(rows: list[dict[str, object]]) -> Path:
    output_path = PROJECT_ROOT / "results" / "roi_phase2_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "model",
        "seed",
        "best_epoch",
        "best_acc",
        "final_epoch",
        "final_acc",
        "macro_f1",
        "early_stopped",
        "best_ckpt",
        "log_path",
        "checkpoint_size_bytes",
        "mean_epoch_time_sec",
        "mean_checkpoint_time_sec",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def print_table(rows: list[dict[str, object]], output_path: Path) -> None:
    print("Phase 2 ROI summary")
    print(f"Saved: {output_path}")
    print(
        f"{'run_name':36} {'model':22} {'best_epoch':>10} {'best_acc':>9} "
        f"{'final_epoch':>11} {'final_acc':>9} {'macro_f1':>9} {'early':>6}"
    )
    for row in rows:
        print(
            f"{str(row['run_name'])[:36]:36} {str(row['model'])[:22]:22} "
            f"{int(row['best_epoch']):10d} {float(row['best_acc']):9.6f} "
            f"{int(row['final_epoch']):11d} {float(row['final_acc']):9.6f} "
            f"{float(row['macro_f1']):9.6f} {str(row['early_stopped']):>6}"
        )

    baseline = next(row for row in rows if row["run_name"] == BASELINE_RUN)
    baseline_acc = float(baseline["best_acc"])
    print()
    print(f"clip baseline best_acc = {baseline_acc:.6f}")
    best_model = max(rows, key=lambda row: float(row["best_acc"]))
    for run_name, label in [
        ("roi_frequency_only_seed42", "frequency_only"),
        ("roi_clip_fft_concat_seed42", "clip_fft_concat"),
        ("roi_affectspectrum_film_seed42", "affectspectrum_film"),
    ]:
        row = next(row for row in rows if row["run_name"] == run_name)
        acc = float(row["best_acc"])
        delta = acc - baseline_acc
        print(f"{label} best_acc = {acc:.6f} | delta_vs_baseline = {delta:+.6f}")
    film = next(row for row in rows if row["run_name"] == "roi_affectspectrum_film_seed42")
    film_delta = float(film["best_acc"]) - baseline_acc
    print(f"current best model = {best_model['model']} ({best_model['run_name']}) best_acc={float(best_model['best_acc']):.6f}")
    print(f"affectspectrum_film exceeds baseline: {film_delta > 0.0}")
    print(f"affectspectrum_film within 1% absolute of baseline: {abs(film_delta) <= 0.01}")


def main() -> int:
    rows = [summarize_run(run_name, model_name) for run_name, model_name in RUNS]
    output_path = write_summary(rows)
    print_table(rows, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
