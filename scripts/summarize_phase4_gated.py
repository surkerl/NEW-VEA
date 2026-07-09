from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GATED_RUNS = [f"roi_affectspectrum_gated_seed{seed}" for seed in (42, 43, 44)]
BASELINE_RUNS = {seed: f"roi_clip_baseline_lr5e4_seed{seed}" for seed in (42, 43, 44)}
CONCAT_RUNS = {seed: f"roi_clip_fft_concat_seed{seed}" for seed in (42, 43, 44)}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 4 input is missing: {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def seed_from_run(run_name: str) -> int:
    match = re.search(r"seed(\d+)$", run_name)
    if not match:
        raise ValueError(f"Cannot infer seed from run name: {run_name}")
    return int(match.group(1))


def read_macro_f1(run_name: str) -> float:
    text = require_file(PROJECT_ROOT / "results" / run_name / "eval_report.txt").read_text(encoding="utf-8")
    match = re.search(r"^macro_f1:\s*([0-9.]+)", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"macro_f1 not found for {run_name}")
    return float(match.group(1))


def early_stopped(run_name: str) -> bool:
    log_path = require_file(PROJECT_ROOT / "logs" / f"{run_name}.log")
    return "Early stopping triggered" in log_path.read_text(encoding="utf-8", errors="replace")


def numeric_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field, "")
        if value not in ("", None):
            values.append(float(value))
    return values


def summarize_gated_run(run_name: str) -> dict[str, object]:
    seed = seed_from_run(run_name)
    rows = read_csv(PROJECT_ROOT / "results" / run_name / "metrics.csv")
    if not rows:
        raise RuntimeError(f"Empty metrics for {run_name}")
    best_row = max(rows, key=lambda row: float(row["test_acc"]))
    final_row = rows[-1]
    ckpt_path = PROJECT_ROOT / "checkpoints" / run_name / "best.pt"
    require_file(ckpt_path)
    return {
        "run_name": run_name,
        "model": "affectspectrum_gated",
        "seed": seed,
        "best_epoch": int(best_row["epoch"]),
        "best_acc": float(best_row["test_acc"]),
        "final_epoch": int(final_row["epoch"]),
        "final_acc": float(final_row["test_acc"]),
        "macro_f1": read_macro_f1(run_name),
        "early_stopped": early_stopped(run_name),
        "best_ckpt": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "checkpoint_size_bytes": ckpt_path.stat().st_size,
        "mean_epoch_time_sec": mean([float(row.get("epoch_elapsed_sec", row.get("elapsed_sec", 0.0))) for row in rows]),
        "mean_checkpoint_time_sec": mean([float(row.get("checkpoint_elapsed_sec", 0.0)) for row in rows]),
        "mean_train_gate": mean(numeric_values(rows, "train_gate_mean")),
        "mean_test_gate": mean(numeric_values(rows, "test_gate_mean")),
    }


def phase2_by_seed(model: str) -> dict[int, dict[str, str]]:
    rows = read_csv(PROJECT_ROOT / "results" / "roi_phase2_fairness_summary.csv")
    output = {}
    for row in rows:
        if row["model"] == model:
            output[int(row["seed"])] = row
    for seed in (42, 43, 44):
        if seed not in output:
            raise RuntimeError(f"Phase 2.5 summary missing {model} seed {seed}")
    return output


def aggregate_gated(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline = phase2_by_seed("baseline_lr5e4")
    concat = phase2_by_seed("clip_fft_concat")
    deltas_baseline = [
        float(row["best_acc"]) - float(baseline[int(row["seed"])]["best_acc"])
        for row in summary_rows
    ]
    deltas_concat = [
        float(row["best_acc"]) - float(concat[int(row["seed"])]["best_acc"])
        for row in summary_rows
    ]
    return [
        {
            "model": "affectspectrum_gated",
            "num_seeds": len(summary_rows),
            "mean_best_acc": mean([float(row["best_acc"]) for row in summary_rows]),
            "std_best_acc": std([float(row["best_acc"]) for row in summary_rows]),
            "mean_macro_f1": mean([float(row["macro_f1"]) for row in summary_rows]),
            "std_macro_f1": std([float(row["macro_f1"]) for row in summary_rows]),
            "mean_best_epoch": mean([float(row["best_epoch"]) for row in summary_rows]),
            "win_count_vs_baseline_lr5e4": sum(delta > 0.0 for delta in deltas_baseline),
            "win_count_vs_concat": sum(delta > 0.0 for delta in deltas_concat),
            "mean_delta_vs_baseline_lr5e4": mean(deltas_baseline),
            "mean_delta_vs_concat": mean(deltas_concat),
            "mean_gate": mean([float(row["mean_test_gate"]) for row in summary_rows]),
        }
    ]


def read_predictions(run_name: str) -> dict[str, dict[str, object]]:
    rows = read_csv(PROJECT_ROOT / "results" / run_name / "predictions.csv")
    output = {}
    for row in rows:
        path = row["path"]
        label = int(float(row.get("label", row.get("y_true", 0))))
        pred = int(float(row.get("pred", row.get("y_pred", 0))))
        correct = row.get("correct")
        is_correct = (str(correct).lower() in {"1", "true", "yes"}) if correct is not None else label == pred
        output[path] = {
            "path": path,
            "label": label,
            "pred": pred,
            "correct": is_correct,
            "class_name": row.get("class_name", f"class_{label}") or f"class_{label}",
        }
    return output


def overlap_counts(reference: dict[str, dict[str, object]], gated: dict[str, dict[str, object]]) -> dict[str, int]:
    if set(reference) != set(gated):
        raise RuntimeError("Prediction sample sets differ for error overlap.")
    counts = {
        "both_correct": 0,
        "reference_correct_gated_wrong": 0,
        "reference_wrong_gated_correct": 0,
        "both_wrong_same_pred": 0,
        "both_wrong_different_pred": 0,
    }
    for path, ref in reference.items():
        gate = gated[path]
        if ref["correct"] and gate["correct"]:
            counts["both_correct"] += 1
        elif ref["correct"] and not gate["correct"]:
            counts["reference_correct_gated_wrong"] += 1
        elif not ref["correct"] and gate["correct"]:
            counts["reference_wrong_gated_correct"] += 1
        elif ref["pred"] == gate["pred"]:
            counts["both_wrong_same_pred"] += 1
        else:
            counts["both_wrong_different_pred"] += 1
    return counts


def build_error_overlap() -> list[dict[str, object]]:
    rows = []
    for seed in (42, 43, 44):
        gated = read_predictions(f"roi_affectspectrum_gated_seed{seed}")
        for comparison, reference_run in (
            ("baseline_vs_gated", BASELINE_RUNS[seed]),
            ("concat_vs_gated", CONCAT_RUNS[seed]),
        ):
            reference = read_predictions(reference_run)
            counts = overlap_counts(reference, gated)
            rows.append(
                {
                    "seed": seed,
                    "comparison": comparison,
                    **counts,
                    "net_corrections": counts["reference_wrong_gated_correct"] - counts["reference_correct_gated_wrong"],
                }
            )
    return rows


def class_accuracy(predictions: dict[str, dict[str, object]], class_idx: int) -> tuple[int, float, str]:
    samples = [row for row in predictions.values() if int(row["label"]) == class_idx]
    class_name = samples[0]["class_name"] if samples else f"class_{class_idx}"
    acc = sum(bool(row["correct"]) for row in samples) / max(len(samples), 1)
    return len(samples), acc, str(class_name)


def build_per_class_delta() -> list[dict[str, object]]:
    rows = []
    for seed in (42, 43, 44):
        baseline = read_predictions(BASELINE_RUNS[seed])
        concat = read_predictions(CONCAT_RUNS[seed])
        gated = read_predictions(f"roi_affectspectrum_gated_seed{seed}")
        classes = sorted({int(row["label"]) for row in baseline.values()})
        for class_idx in classes:
            _, baseline_acc, class_name = class_accuracy(baseline, class_idx)
            _, concat_acc, _ = class_accuracy(concat, class_idx)
            _, gated_acc, _ = class_accuracy(gated, class_idx)
            rows.append(
                {
                    "seed": seed,
                    "class_idx": class_idx,
                    "class_name": class_name,
                    "baseline_acc": baseline_acc,
                    "concat_acc": concat_acc,
                    "gated_acc": gated_acc,
                    "gated_delta_vs_baseline": gated_acc - baseline_acc,
                    "gated_delta_vs_concat": gated_acc - concat_acc,
                }
            )
    return rows


def build_probe_comparison() -> list[dict[str, object]]:
    phase3_path = PROJECT_ROOT / "results" / "phase3b_spectral_probe_model_comparison_mean.csv"
    phase4_dir = PROJECT_ROOT / "results" / "phase4_gated_spectral_probe"
    if not phase3_path.exists() or not phase4_dir.exists():
        return []

    phase3_rows = {row["perturbation"]: row for row in read_csv(phase3_path)}
    gated_by_perturbation: dict[str, list[float]] = {}
    for metrics_path in sorted(phase4_dir.glob("roi_affectspectrum_gated_seed*/spectral_probe_metrics.csv")):
        for row in read_csv(metrics_path):
            gated_by_perturbation.setdefault(row["perturbation"], []).append(float(row["acc"]))
    if not gated_by_perturbation:
        raise RuntimeError(f"No gated spectral probe metrics found under {phase4_dir}")

    gated_full = mean(gated_by_perturbation.get("full", []))
    output = []
    for perturbation, values in sorted(gated_by_perturbation.items()):
        if perturbation not in phase3_rows:
            raise RuntimeError(f"Phase 3 probe comparison missing perturbation: {perturbation}")
        phase3 = phase3_rows[perturbation]
        gated_mean = mean(values)
        output.append(
            {
                "perturbation": perturbation,
                "baseline_mean_acc": float(phase3["baseline_mean_acc"]),
                "concat_mean_acc": float(phase3["concat_mean_acc"]),
                "film_mean_acc": float(phase3["film_mean_acc"]),
                "gated_mean_acc": gated_mean,
                "gated_delta_vs_baseline": gated_mean - float(phase3["baseline_mean_acc"]),
                "gated_delta_vs_concat": gated_mean - float(phase3["concat_mean_acc"]),
                "gated_drop_vs_full": gated_full - gated_mean,
            }
        )
    return output


def summarize_response_loss(summary_rows: list[dict[str, object]]) -> str:
    parts = []
    for row in summary_rows:
        metrics = read_csv(PROJECT_ROOT / "results" / str(row["run_name"]) / "metrics.csv")
        response_values = numeric_values(metrics, "train_response_loss")
        if len(response_values) >= 2:
            parts.append(
                f"seed{row['seed']}: train_response_loss {response_values[0]:.4f}->{response_values[-1]:.4f}"
            )
    return "; ".join(parts) if parts else "response loss not recorded"


def write_markdown(
    summary_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
    overlap_rows: list[dict[str, object]],
    probe_rows: list[dict[str, object]],
) -> None:
    aggregate = aggregate_rows[0]
    mean_gate = float(aggregate["mean_gate"])
    baseline_overlap = [row for row in overlap_rows if row["comparison"] == "baseline_vs_gated"]
    concat_overlap = [row for row in overlap_rows if row["comparison"] == "concat_vs_gated"]
    stronger_than_concat = [row for row in probe_rows if float(row["gated_delta_vs_concat"]) > 0.0 and row["perturbation"] != "full"]
    weaker_than_concat = [row for row in probe_rows if float(row["gated_delta_vs_concat"]) < 0.0 and row["perturbation"] != "full"]
    low_blur_down = [
        row for row in probe_rows
        if str(row["perturbation"]).startswith("low_")
        or str(row["perturbation"]).startswith("blur_")
        or str(row["perturbation"]).startswith("downsample_")
    ]
    lines = [
        "# Phase 4 AffectSpectrum-Gated v1 Summary",
        "",
        f"Gated mean best_acc: {float(aggregate['mean_best_acc']):.6f}",
        f"Gated mean macro_f1: {float(aggregate['mean_macro_f1']):.6f}",
        f"Mean delta vs baseline_lr5e4: {float(aggregate['mean_delta_vs_baseline_lr5e4']):+.6f}",
        f"Mean delta vs concat: {float(aggregate['mean_delta_vs_concat']):+.6f}",
        f"Win count vs baseline_lr5e4: {aggregate['win_count_vs_baseline_lr5e4']}/3",
        f"Win count vs concat: {aggregate['win_count_vs_concat']}/3",
        "",
        "Per-seed clean results:",
    ]
    for row in summary_rows:
        lines.append(
            f"- seed{row['seed']}: best_acc={float(row['best_acc']):.6f}, "
            f"best_epoch={row['best_epoch']}, macro_f1={float(row['macro_f1']):.6f}, "
            f"mean_test_gate={float(row['mean_test_gate']):.4f}"
        )
    lines.extend(
        [
            "",
            f"Baseline vs gated mean net correction: {mean([float(row['net_corrections']) for row in baseline_overlap]):+.2f}",
            f"Concat vs gated mean net correction: {mean([float(row['net_corrections']) for row in concat_overlap]):+.2f}",
            "",
            f"Mean gate: {mean_gate:.4f}. "
            + ("Gate is not collapsed to 0 or 1." if 0.1 <= mean_gate <= 0.9 else "Gate is close to a boundary and should be inspected."),
            f"Response auxiliary loss trend: {summarize_response_loss(summary_rows)}.",
            "",
            "Gated stronger than concat under: "
            + (", ".join(f"{row['perturbation']} ({float(row['gated_delta_vs_concat']):+.4f})" for row in stronger_than_concat) or "none"),
            "",
            "Gated weaker than concat under: "
            + (", ".join(f"{row['perturbation']} ({float(row['gated_delta_vs_concat']):+.4f})" for row in weaker_than_concat) or "none"),
            "",
            "Low/blur/downsample comparison vs concat: "
            + (
                ", ".join(
                    f"{row['perturbation']} ({float(row['gated_delta_vs_concat']):+.4f})"
                    for row in low_blur_down
                )
                if low_blur_down
                else "phase4 spectral probe unavailable"
            ),
            "",
            "Recommendation: proceed to Phase 5 only if the clean and perturbation summaries show a useful tradeoff against concat; otherwise tune the gate regularization and response loss before adding log-spaced spectrum or local spectral tokens.",
            "",
        ]
    )
    (PROJECT_ROOT / "results" / "phase4_gated_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    require_file(PROJECT_ROOT / "results" / "roi_phase2_fairness_summary.csv")
    require_file(PROJECT_ROOT / "results" / "roi_phase2_fairness_aggregate.csv")
    summary_rows = [summarize_gated_run(run_name) for run_name in GATED_RUNS]
    aggregate_rows = aggregate_gated(summary_rows)
    overlap_rows = build_error_overlap()
    per_class_rows = build_per_class_delta()
    probe_rows = build_probe_comparison()

    write_csv(PROJECT_ROOT / "results" / "phase4_gated_summary.csv", summary_rows)
    write_csv(PROJECT_ROOT / "results" / "phase4_gated_aggregate.csv", aggregate_rows)
    write_csv(PROJECT_ROOT / "results" / "phase4_gated_error_overlap.csv", overlap_rows)
    write_csv(PROJECT_ROOT / "results" / "phase4_gated_per_class_delta.csv", per_class_rows)
    write_csv(PROJECT_ROOT / "results" / "phase4_gated_probe_comparison.csv", probe_rows)
    write_markdown(summary_rows, aggregate_rows, overlap_rows, probe_rows)

    print("Phase 4 gated per-run")
    for row in summary_rows:
        print(
            f"seed{row['seed']} best_acc={float(row['best_acc']):.6f} "
            f"best_epoch={row['best_epoch']} macro_f1={float(row['macro_f1']):.6f} "
            f"mean_gate={float(row['mean_test_gate']):.4f}"
        )
    print("Phase 4 gated aggregate")
    print(aggregate_rows[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
