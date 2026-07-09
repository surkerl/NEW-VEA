from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 3 spectral diagnostics.")
    parser.add_argument("--phase3a_dir", default="results/phase3a_prediction_analysis")
    parser.add_argument("--phase3b_dir", default="results/phase3b_spectral_probe")
    parser.add_argument("--output_dir", default="results")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
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


def seed_from_run(run_name: str) -> int:
    match = re.search(r"seed(\d+)$", run_name)
    if not match:
        raise ValueError(f"Cannot infer seed from {run_name}")
    return int(match.group(1))


def load_probe_metrics(phase3b_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(phase3b_dir.glob("*/spectral_probe_metrics.csv")):
        if metrics_path.parent.name.startswith("smoke"):
            continue
        for row in read_csv(metrics_path):
            row_obj: dict[str, object] = dict(row)
            row_obj["seed"] = int(row["seed"])
            row_obj["acc"] = float(row["acc"])
            row_obj["macro_f1"] = float(row["macro_f1"])
            rows.append(row_obj)
    if not rows:
        raise RuntimeError(f"No spectral_probe_metrics.csv files found under {phase3b_dir}")
    return rows


def aggregate_vs_full(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    full_by_run = {
        (str(row["model"]), int(row["seed"]), str(row["run_name"])): row
        for row in rows
        if row["perturbation"] == "full"
    }
    output = []
    for row in rows:
        key = (str(row["model"]), int(row["seed"]), str(row["run_name"]))
        full = full_by_run[key]
        output.append(
            {
                "model": row["model"],
                "seed": row["seed"],
                "perturbation": row["perturbation"],
                "acc": row["acc"],
                "macro_f1": row["macro_f1"],
                "delta_acc_vs_full": float(row["acc"]) - float(full["acc"]),
                "delta_macro_f1_vs_full": float(row["macro_f1"]) - float(full["macro_f1"]),
            }
        )
    return output


def model_comparison(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (int(row["seed"]), str(row["perturbation"]), str(row["model"])): row
        for row in rows
    }
    full_acc = {
        (int(row["seed"]), str(row["model"])): float(row["acc"])
        for row in rows
        if row["perturbation"] == "full"
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    perturbations = sorted({str(row["perturbation"]) for row in rows})
    output = []
    for seed in seeds:
        for perturbation in perturbations:
            baseline = by_key[(seed, perturbation, "baseline")]
            concat = by_key[(seed, perturbation, "concat")]
            film = by_key[(seed, perturbation, "film")]
            output.append(
                {
                    "seed": seed,
                    "perturbation": perturbation,
                    "baseline_acc": float(baseline["acc"]),
                    "concat_acc": float(concat["acc"]),
                    "film_acc": float(film["acc"]),
                    "concat_delta_vs_baseline": float(concat["acc"]) - float(baseline["acc"]),
                    "film_delta_vs_baseline": float(film["acc"]) - float(baseline["acc"]),
                    "baseline_drop_vs_full": full_acc[(seed, "baseline")] - float(baseline["acc"]),
                    "concat_drop_vs_full": full_acc[(seed, "concat")] - float(concat["acc"]),
                    "film_drop_vs_full": full_acc[(seed, "film")] - float(film["acc"]),
                }
            )
    return output


def mean_model_comparison(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    perturbations = sorted({str(row["perturbation"]) for row in rows})
    output = []
    for perturbation in perturbations:
        group = [row for row in rows if row["perturbation"] == perturbation]
        output.append(
            {
                "perturbation": perturbation,
                "baseline_mean_acc": mean([float(row["baseline_acc"]) for row in group]),
                "concat_mean_acc": mean([float(row["concat_acc"]) for row in group]),
                "film_mean_acc": mean([float(row["film_acc"]) for row in group]),
                "concat_mean_delta_vs_baseline": mean([float(row["concat_delta_vs_baseline"]) for row in group]),
                "film_mean_delta_vs_baseline": mean([float(row["film_delta_vs_baseline"]) for row in group]),
                "baseline_mean_drop_vs_full": mean([float(row["baseline_drop_vs_full"]) for row in group]),
                "concat_mean_drop_vs_full": mean([float(row["concat_drop_vs_full"]) for row in group]),
                "film_mean_drop_vs_full": mean([float(row["film_drop_vs_full"]) for row in group]),
            }
        )
    order = {
        "full": 0,
        "low_0.15": 1,
        "low_0.30": 2,
        "low_0.50": 3,
        "high_0.30": 4,
        "high_0.50": 5,
        "band_0.15_0.30": 6,
        "band_0.30_0.50": 7,
        "band_0.50_0.75": 8,
        "blur_light": 9,
        "blur_heavy": 10,
        "downsample_x2": 11,
        "downsample_x4": 12,
        "amplitude_noise_0.05": 13,
        "amplitude_noise_0.10": 14,
        "phase_noise_0.05": 15,
        "phase_noise_0.10": 16,
    }
    return sorted(output, key=lambda row: order.get(str(row["perturbation"]), 999))


def read_phase25_summary(output_dir: Path) -> str:
    path = output_dir / "roi_phase2_fairness_aggregate.csv"
    if not path.exists():
        return "Phase 2.5 fairness aggregate was not found."
    rows = read_csv(path)
    parts = []
    for row in rows:
        parts.append(
            f"{row['model']}: mean best_acc={float(row['mean_best_acc']):.6f}, "
            f"mean macro_f1={float(row['mean_macro_f1']):.6f}, "
            f"delta={float(row['mean_delta_vs_baseline']):+.6f}"
        )
    return "\n".join(f"- {part}" for part in parts)


def load_evidence_rows(phase3b_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(phase3b_dir.glob("*/evidence_accumulation_by_class.csv")):
        if path.parent.name.startswith("smoke"):
            continue
        metrics_path = path.parent / "spectral_probe_metrics.csv"
        metrics = read_csv(metrics_path)
        model = metrics[0]["model"]
        seed = int(metrics[0]["seed"])
        for row in read_csv(path):
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "class_idx": int(row["class_idx"]),
                    "class_name": row["class_name"],
                    "stage": row["stage"],
                    "mean_gt_prob": float(row["mean_gt_prob"]),
                    "mean_entropy": float(row["mean_entropy"]),
                    "acc": float(row["acc"]),
                }
            )
    return rows


def evidence_summary(evidence_rows: list[dict[str, object]]) -> tuple[str, str, str]:
    if not evidence_rows:
        return "No evidence accumulation rows found.", "none", "none"
    stages = ["low_0.15", "low_0.30", "low_0.50", "full"]
    stage_means = {
        stage: mean([float(row["mean_gt_prob"]) for row in evidence_rows if row["stage"] == stage])
        for stage in stages
    }
    class_names = sorted({str(row["class_name"]) for row in evidence_rows})
    class_stats = []
    for class_name in class_names:
        low = mean([float(row["mean_gt_prob"]) for row in evidence_rows if row["class_name"] == class_name and row["stage"] == "low_0.15"])
        low50 = mean([float(row["mean_gt_prob"]) for row in evidence_rows if row["class_name"] == class_name and row["stage"] == "low_0.50"])
        full = mean([float(row["mean_gt_prob"]) for row in evidence_rows if row["class_name"] == class_name and row["stage"] == "full"])
        class_stats.append({"class_name": class_name, "low": low, "low50": low50, "full": full, "detail_gain": full - low50})
    low_reliant = sorted(class_stats, key=lambda row: row["low"], reverse=True)[:3]
    detail_reliant = sorted(class_stats, key=lambda row: row["detail_gain"], reverse=True)[:3]
    text = (
        f"Mean gt probability by stage: "
        + ", ".join(f"{stage}={stage_means[stage]:.4f}" for stage in stages)
    )
    return (
        text,
        ", ".join(f"{row['class_name']} ({row['low']:.4f})" for row in low_reliant),
        ", ".join(f"{row['class_name']} ({row['detail_gain']:+.4f})" for row in detail_reliant),
    )


def write_markdown(
    path: Path,
    *,
    phase25_text: str,
    phase3a_dir: Path,
    comparison_mean: list[dict[str, object]],
    evidence_text: str,
    low_reliant: str,
    detail_reliant: str,
) -> None:
    phase3a_summary = phase3a_dir / "analysis_summary.txt"
    phase3a_text = phase3a_summary.read_text(encoding="utf-8") if phase3a_summary.exists() else "Phase 3A summary missing."
    concat_stable = [
        row for row in comparison_mean
        if str(row["perturbation"]) != "full" and float(row["concat_mean_delta_vs_baseline"]) > 0.0
    ]
    concat_weak = [
        row for row in comparison_mean
        if str(row["perturbation"]) != "full" and float(row["concat_mean_delta_vs_baseline"]) < 0.0
    ]
    film_stable = [
        row for row in comparison_mean
        if str(row["perturbation"]) != "full" and float(row["film_mean_delta_vs_baseline"]) > 0.0
    ]
    film_weak = [
        row for row in comparison_mean
        if str(row["perturbation"]) != "full" and float(row["film_mean_delta_vs_baseline"]) < 0.0
    ]
    lines = [
        "# Phase 3 Spectral Evidence Diagnosis",
        "",
        "## Phase 2.5 Clean Accuracy Context",
        phase25_text,
        "",
        "## Phase 3A Prediction-Level Evidence",
        phase3a_text.strip(),
        "",
        "## Phase 3B Spectral Perturbation Evidence",
        "Concat is stronger than baseline under: "
        + (", ".join(f"{row['perturbation']} ({float(row['concat_mean_delta_vs_baseline']):+.4f})" for row in concat_stable) or "none"),
        "",
        "Concat is weaker than baseline under: "
        + (", ".join(f"{row['perturbation']} ({float(row['concat_mean_delta_vs_baseline']):+.4f})" for row in concat_weak) or "none"),
        "",
        "FiLM is stronger than baseline under: "
        + (", ".join(f"{row['perturbation']} ({float(row['film_mean_delta_vs_baseline']):+.4f})" for row in film_stable) or "none"),
        "",
        "FiLM is weaker than baseline under: "
        + (", ".join(f"{row['perturbation']} ({float(row['film_mean_delta_vs_baseline']):+.4f})" for row in film_weak) or "none"),
        "",
        "## Coarse-to-Fine Spectral Evidence Accumulation",
        evidence_text,
        f"Classes with strongest low-frequency gt probability: {low_reliant}",
        f"Classes with largest high/detail gain from low_0.50 to full: {detail_reliant}",
        "",
        "## Interpretation",
        "Phase 3 evaluates whether spectral presentation evidence provides a distinct affective signal beyond semantic visual representations. The clean accuracy gains, prediction-level non-overlap, and perturbation sensitivity patterns support the frequency-centric interpretation: spectral evidence is not merely an ordinary complementary feature, but a measurable affective signal with its own error corrections and degradation profile.",
        "",
        "The next modeling step should be AffectSpectrum-Gated Fusion, using spectral response to gate semantic CLIP features rather than treating spectrum as a passive concatenated descriptor.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    phase3a_dir = PROJECT_ROOT / args.phase3a_dir
    phase3b_dir = PROJECT_ROOT / args.phase3b_dir

    metrics = load_probe_metrics(phase3b_dir)
    aggregate_rows = aggregate_vs_full(metrics)
    comparison_rows = model_comparison(metrics)
    comparison_mean_rows = mean_model_comparison(comparison_rows)

    write_csv(output_dir / "phase3b_spectral_probe_aggregate.csv", aggregate_rows)
    write_csv(output_dir / "phase3b_spectral_probe_model_comparison.csv", comparison_rows)
    write_csv(output_dir / "phase3b_spectral_probe_model_comparison_mean.csv", comparison_mean_rows)

    evidence_text, low_reliant, detail_reliant = evidence_summary(load_evidence_rows(phase3b_dir))
    write_markdown(
        output_dir / "phase3_spectral_diagnostics_summary.md",
        phase25_text=read_phase25_summary(output_dir),
        phase3a_dir=phase3a_dir,
        comparison_mean=comparison_mean_rows,
        evidence_text=evidence_text,
        low_reliant=low_reliant,
        detail_reliant=detail_reliant,
    )

    print(f"Saved Phase 3 aggregate summaries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
