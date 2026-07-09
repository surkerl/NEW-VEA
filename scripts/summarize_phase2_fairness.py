import csv
import re
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch


RUN_GROUPS = {
    "baseline_lr5e4": [f"roi_clip_baseline_lr5e4_seed{seed}" for seed in (42, 43, 44)],
    "clip_fft_concat": [f"roi_clip_fft_concat_seed{seed}" for seed in (42, 43, 44)],
    "affectspectrum_film": [f"roi_affectspectrum_film_seed{seed}" for seed in (42, 43, 44)],
}


def read_metrics(run_name: str) -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / run_name / "metrics.csv"
    with path.open(newline="", encoding="utf-8") as f:
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


def model_label_from_run(run_name: str) -> str:
    if "baseline_lr5e4" in run_name:
        return "baseline_lr5e4"
    if "clip_fft_concat" in run_name:
        return "clip_fft_concat"
    if "affectspectrum_film" in run_name:
        return "affectspectrum_film"
    raise ValueError(f"Cannot infer model label from {run_name}")


def seed_from_run(run_name: str) -> int:
    match = re.search(r"seed(\d+)$", run_name)
    if not match:
        raise ValueError(f"Cannot infer seed from {run_name}")
    return int(match.group(1))


def summarize_run(run_name: str) -> dict[str, object]:
    rows = read_metrics(run_name)
    best_row = max(rows, key=lambda row: float(row["test_acc"]))
    final_row = rows[-1]
    ckpt_path = PROJECT_ROOT / "checkpoints" / run_name / "best.pt"
    mean_epoch_time = sum(float(row.get("epoch_elapsed_sec", row.get("elapsed_sec", 0.0))) for row in rows) / len(rows)
    mean_checkpoint_time = sum(float(row.get("checkpoint_elapsed_sec", 0.0)) for row in rows) / len(rows)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config_seed = checkpoint.get("config", {}).get("train", {}).get("seed")

    return {
        "run_name": run_name,
        "model": model_label_from_run(run_name),
        "seed": int(config_seed) if config_seed is not None else seed_from_run(run_name),
        "best_epoch": int(best_row["epoch"]),
        "best_acc": float(best_row["test_acc"]),
        "final_epoch": int(final_row["epoch"]),
        "final_acc": float(final_row["test_acc"]),
        "macro_f1": read_macro_f1(run_name),
        "early_stopped": early_stopped(run_name),
        "checkpoint_size_bytes": ckpt_path.stat().st_size,
        "mean_epoch_time_sec": mean_epoch_time,
        "mean_checkpoint_time_sec": mean_checkpoint_time,
    }


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    baseline_by_seed = {
        int(row["seed"]): float(row["best_acc"])
        for row in by_model["baseline_lr5e4"]
    }
    output = []
    for model, model_rows in by_model.items():
        best_accs = [float(row["best_acc"]) for row in model_rows]
        macro_f1s = [float(row["macro_f1"]) for row in model_rows]
        best_epochs = [float(row["best_epoch"]) for row in model_rows]
        deltas = [
            float(row["best_acc"]) - baseline_by_seed[int(row["seed"])]
            for row in model_rows
        ]
        output.append(
            {
                "model": model,
                "num_seeds": len(model_rows),
                "mean_best_acc": statistics.mean(best_accs),
                "std_best_acc": std(best_accs),
                "mean_macro_f1": statistics.mean(macro_f1s),
                "std_macro_f1": std(macro_f1s),
                "mean_best_epoch": statistics.mean(best_epochs),
                "win_count_vs_baseline": sum(delta > 0.0 for delta in deltas) if model != "baseline_lr5e4" else 0,
                "mean_delta_vs_baseline": statistics.mean(deltas) if model != "baseline_lr5e4" else 0.0,
            }
        )
    order = {"baseline_lr5e4": 0, "clip_fft_concat": 1, "affectspectrum_film": 2}
    return sorted(output, key=lambda row: order[str(row["model"])])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_tables(rows: list[dict[str, object]], aggregate_rows: list[dict[str, object]]) -> None:
    print("Phase 2.5 fairness per-run")
    print(f"{'run_name':36} {'model':22} {'seed':>4} {'best_epoch':>10} {'best_acc':>9} {'macro_f1':>9} {'early':>6}")
    for row in rows:
        print(
            f"{str(row['run_name'])[:36]:36} {str(row['model'])[:22]:22} "
            f"{int(row['seed']):4d} {int(row['best_epoch']):10d} "
            f"{float(row['best_acc']):9.6f} {float(row['macro_f1']):9.6f} {str(row['early_stopped']):>6}"
        )
    print()
    print("Phase 2.5 fairness aggregate")
    print(
        f"{'model':22} {'n':>3} {'mean_acc':>9} {'std_acc':>9} "
        f"{'mean_f1':>9} {'std_f1':>9} {'wins':>5} {'mean_delta':>11}"
    )
    for row in aggregate_rows:
        print(
            f"{str(row['model'])[:22]:22} {int(row['num_seeds']):3d} "
            f"{float(row['mean_best_acc']):9.6f} {float(row['std_best_acc']):9.6f} "
            f"{float(row['mean_macro_f1']):9.6f} {float(row['std_macro_f1']):9.6f} "
            f"{int(row['win_count_vs_baseline']):5d} {float(row['mean_delta_vs_baseline']):11.6f}"
        )


def main() -> int:
    run_names = [run for group in RUN_GROUPS.values() for run in group]
    rows = [summarize_run(run_name) for run_name in run_names]
    aggregate_rows = aggregate(rows)
    write_csv(PROJECT_ROOT / "results" / "roi_phase2_fairness_summary.csv", rows)
    write_csv(PROJECT_ROOT / "results" / "roi_phase2_fairness_aggregate.csv", aggregate_rows)
    print_tables(rows, aggregate_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
