from __future__ import annotations

import csv
import math
import re
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUNS = {
    "baseline": "roi_clip_baseline_lr5e4_seed42",
    "concat": "roi_clip_fft_concat_seed42",
    "gated": "roi_affectspectrum_gated_seed42",
}
PHASE7_RUNS = {
    "spatial_token_adapter": ("spatial", "roi_spatial_token_adapter_seed42"),
    "spectral_global_filter_adapter": ("global_filter", "roi_spectral_global_filter_adapter_seed42"),
    "spectral_factorized_filter_adapter": (
        "factorized_filter",
        "roi_spectral_factorized_filter_adapter_seed42",
    ),
    "wavelet_token_adapter": ("wavelet", "roi_wavelet_token_adapter_seed42"),
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 7 input is missing: {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with require_file(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_numeric(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return statistics.mean(values) if values else 0.0


def best_acc(run_name: str) -> float:
    rows = read_csv(PROJECT_ROOT / "results" / run_name / "metrics.csv")
    if not rows:
        raise RuntimeError(f"Empty metrics.csv for {run_name}")
    return max(float(row["test_acc"]) for row in rows)


def macro_f1(run_name: str) -> float:
    text = require_file(PROJECT_ROOT / "results" / run_name / "eval_report.txt").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^macro_f1:\s*([0-9.]+)", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"macro_f1 not found in eval_report.txt for {run_name}")
    return float(match.group(1))


def prediction_diagnostic(run_name: str, fields: list[str]) -> float:
    rows = read_csv(PROJECT_ROOT / "results" / run_name / "predictions.csv")
    values = [
        abs(float(row[field]))
        for row in rows
        for field in fields
        if row.get(field) not in (None, "")
    ]
    return statistics.mean(values) if values else 0.0


def parameter_counts() -> dict[str, dict[str, str]]:
    rows = read_csv(PROJECT_ROOT / "results" / "phase7_parameter_counts.csv")
    counts = {row["model"]: row for row in rows}
    missing = set(PHASE7_RUNS) - set(counts)
    if missing:
        raise RuntimeError(f"phase7_parameter_counts.csv missing models: {sorted(missing)}")
    classifier_counts = {int(counts[model]["classifier_params"]) for model in PHASE7_RUNS}
    if len(classifier_counts) != 1:
        raise RuntimeError("Phase 7 classifier parameter counts differ across candidates.")
    adapter_counts = [int(counts[model]["adapter_params"]) for model in PHASE7_RUNS]
    ratio = max(adapter_counts) / max(min(adapter_counts), 1)
    if ratio > 2.5:
        raise RuntimeError(f"Phase 7 adapter parameter max/min ratio {ratio:.3f} exceeds 2.5.")
    return counts


def summarize_run(
    model_name: str,
    adapter_type: str,
    run_name: str,
    counts: dict[str, str],
    references: dict[str, float],
) -> dict[str, object]:
    rows = read_csv(PROJECT_ROOT / "results" / run_name / "metrics.csv")
    if not rows:
        raise RuntimeError(f"Empty metrics.csv for {run_name}")
    best_row = max(rows, key=lambda row: float(row["test_acc"]))
    final_row = rows[-1]
    log_text = require_file(PROJECT_ROOT / "logs" / f"{run_name}.log").read_text(
        encoding="utf-8", errors="replace"
    )
    checkpoint = require_file(PROJECT_ROOT / "checkpoints" / run_name / "best.pt")
    return {
        "run_name": run_name,
        "model": model_name,
        "adapter_type": adapter_type,
        "seed": 42,
        "best_epoch": int(best_row["epoch"]),
        "best_acc": float(best_row["test_acc"]),
        "final_epoch": int(final_row["epoch"]),
        "final_acc": float(final_row["test_acc"]),
        "macro_f1": macro_f1(run_name),
        "early_stopped": "Early stopping triggered" in log_text,
        "trainable_params": int(counts["trainable_params"]),
        "adapter_params": int(counts["adapter_params"]),
        "classifier_params": int(counts["classifier_params"]),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "mean_epoch_time_sec": mean_numeric(rows, "epoch_elapsed_sec"),
        "mean_adapter_residual_abs_mean": mean_numeric(rows, "test_adapter_residual_abs_mean"),
        "mean_adapter_residual_norm": mean_numeric(rows, "test_adapter_residual_norm"),
        "mean_layer_scale": mean_numeric(rows, "test_layer_scale_mean"),
        "delta_vs_baseline_seed42": float(best_row["test_acc"]) - references["baseline"],
        "delta_vs_concat_seed42": float(best_row["test_acc"]) - references["concat"],
        "delta_vs_gated_seed42": float(best_row["test_acc"]) - references["gated"],
        "delta_vs_spatial_control": 0.0,
    }


def decision(row: dict[str, object], spatial_acc: float) -> str:
    if row["adapter_type"] == "spatial":
        return "spatial_control"
    accuracy = float(row["best_acc"])
    if accuracy >= 0.710000:
        return "strong_continue_3seeds"
    if accuracy >= 0.705387 and accuracy > spatial_acc:
        return "continue_3seeds"
    if accuracy >= 0.700000:
        return "borderline_only_probe_if_interpretable"
    return "stop"


def parse_parity() -> tuple[float, float, bool]:
    text = require_file(PROJECT_ROOT / "results" / "phase7_openclip_visual_inspection.txt").read_text(
        encoding="utf-8"
    )
    max_diff = float(re.search(r"^max_abs_diff:\s*([0-9.eE+-]+)", text, re.MULTILINE).group(1))
    mean_diff = float(re.search(r"^mean_abs_diff:\s*([0-9.eE+-]+)", text, re.MULTILINE).group(1))
    passed = "manual_forward_parity_passed: True" in text
    return max_diff, mean_diff, passed


def main() -> int:
    references = {name: best_acc(run_name) for name, run_name in REFERENCE_RUNS.items()}
    counts = parameter_counts()
    summary_rows = [
        summarize_run(model, adapter_type, run_name, counts[model], references)
        for model, (adapter_type, run_name) in PHASE7_RUNS.items()
    ]
    spatial = next(row for row in summary_rows if row["adapter_type"] == "spatial")
    spatial_acc = float(spatial["best_acc"])
    spatial_f1 = float(spatial["macro_f1"])
    for row in summary_rows:
        row["delta_vs_spatial_control"] = float(row["best_acc"]) - spatial_acc

    ranking_rows = []
    for rank, row in enumerate(sorted(summary_rows, key=lambda item: float(item["best_acc"]), reverse=True), 1):
        ranking_rows.append(
            {
                "rank": rank,
                "model": row["model"],
                "best_acc": row["best_acc"],
                "macro_f1": row["macro_f1"],
                "delta_vs_baseline": row["delta_vs_baseline_seed42"],
                "delta_vs_concat": row["delta_vs_concat_seed42"],
                "delta_vs_spatial_control": row["delta_vs_spatial_control"],
                "decision": decision(row, spatial_acc),
            }
        )

    frequency_rows = [row for row in summary_rows if row["adapter_type"] != "spatial"]
    gain_rows = [
        {
            "frequency_model": row["model"],
            "frequency_best_acc": row["best_acc"],
            "spatial_control_best_acc": spatial_acc,
            "frequency_minus_spatial": float(row["best_acc"]) - spatial_acc,
            "frequency_macro_f1": row["macro_f1"],
            "spatial_control_macro_f1": spatial_f1,
            "frequency_macro_f1_minus_spatial": float(row["macro_f1"]) - spatial_f1,
            "frequency_specific_gain_positive": float(row["best_acc"]) > spatial_acc,
        }
        for row in frequency_rows
    ]

    response_fields = {
        "spectral_global_filter_adapter": ["global_filter_abs_mean"],
        "spectral_factorized_filter_adapter": ["factorized_coeff_abs_mean"],
        "wavelet_token_adapter": [
            "wavelet_ll_scale",
            "wavelet_lh_scale",
            "wavelet_hl_scale",
            "wavelet_hh_scale",
        ],
    }
    response_magnitudes = {
        row["model"]: prediction_diagnostic(str(row["run_name"]), response_fields[str(row["model"])])
        for row in frequency_rows
    }
    best_frequency = max(frequency_rows, key=lambda row: float(row["best_acc"]))
    best_response_nonzero = response_magnitudes[str(best_frequency["model"])] > 1.0e-8
    condition_a = float(best_frequency["best_acc"]) >= 0.710000
    condition_b = (
        float(best_frequency["best_acc"]) >= 0.705387
        and float(best_frequency["best_acc"]) > spatial_acc
    )
    condition_c = (
        float(best_frequency["best_acc"]) >= 0.703000
        and float(best_frequency["best_acc"]) - spatial_acc >= 0.003000
        and float(best_frequency["macro_f1"]) > spatial_f1
        and best_response_nonzero
    )
    verdict = "CONTINUE_FREQUENCY_MAINLINE" if condition_a or condition_b or condition_c else "ABANDON_FREQUENCY_MAINLINE"
    max_diff, mean_diff, parity_passed = parse_parity()

    summary_fields = [
        "run_name", "model", "adapter_type", "seed", "best_epoch", "best_acc",
        "final_epoch", "final_acc", "macro_f1", "early_stopped", "trainable_params",
        "adapter_params", "classifier_params", "checkpoint_size_bytes", "mean_epoch_time_sec",
        "mean_adapter_residual_abs_mean", "mean_adapter_residual_norm", "mean_layer_scale",
        "delta_vs_baseline_seed42", "delta_vs_concat_seed42", "delta_vs_gated_seed42",
        "delta_vs_spatial_control",
    ]
    ranking_fields = [
        "rank", "model", "best_acc", "macro_f1", "delta_vs_baseline",
        "delta_vs_concat", "delta_vs_spatial_control", "decision",
    ]
    gain_fields = [
        "frequency_model", "frequency_best_acc", "spatial_control_best_acc",
        "frequency_minus_spatial", "frequency_macro_f1", "spatial_control_macro_f1",
        "frequency_macro_f1_minus_spatial", "frequency_specific_gain_positive",
    ]
    write_csv(PROJECT_ROOT / "results" / "phase7_internal_adapter_summary.csv", summary_rows, summary_fields)
    write_csv(PROJECT_ROOT / "results" / "phase7_internal_adapter_ranking.csv", ranking_rows, ranking_fields)
    write_csv(PROJECT_ROOT / "results" / "phase7_frequency_specific_gain.csv", gain_rows, gain_fields)

    near_zero_residual = all(float(row["mean_adapter_residual_abs_mean"]) < 1.0e-4 for row in summary_rows)
    layer_scales_near_init = all(
        math.isclose(float(row["mean_layer_scale"]), 1.0e-4, rel_tol=0.25, abs_tol=2.5e-5)
        for row in summary_rows
    )
    lines = [
        "# Phase 7 Internal Spectral Adapter Sprint",
        "",
        f"Manual OpenCLIP parity: {'passed' if parity_passed else 'failed'} "
        f"(max_abs_diff={max_diff:.10e}, mean_abs_diff={mean_diff:.10e}).",
        f"Spatial control best_acc={spatial_acc:.6f}, macro_f1={spatial_f1:.6f}.",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['model']}: best_acc={float(row['best_acc']):.6f}, "
            f"macro_f1={float(row['macro_f1']):.6f}, "
            f"delta_vs_spatial={float(row['delta_vs_spatial_control']):+.6f}, "
            f"mean_residual={float(row['mean_adapter_residual_abs_mean']):.6e}, "
            f"mean_layer_scale={float(row['mean_layer_scale']):.6e}."
        )
    lines.extend(
        [
            "",
            f"Best frequency model: {best_frequency['model']} at {float(best_frequency['best_acc']):.6f}.",
            f"It {'does' if float(best_frequency['best_acc']) > spatial_acc else 'does not'} exceed the spatial control.",
            f"It {'does' if float(best_frequency['best_acc']) >= 0.705387 else 'does not'} reach concat/gated seed42 (0.705387).",
            "Frequency response magnitudes: "
            + ", ".join(f"{model}={value:.6e}" for model, value in response_magnitudes.items())
            + ".",
            f"Adapter residuals are {'near zero' if near_zero_residual else 'non-zero'} by the 1e-4 mean threshold.",
            f"Layer scales are {'still near initialization' if layer_scales_near_init else 'not uniformly near initialization'}.",
            "Frequency-specific gains: "
            + ", ".join(
                f"{row['frequency_model']}={float(row['frequency_minus_spatial']):+.6f}"
                for row in gain_rows
            )
            + ".",
            "",
            f"Decision: {verdict}",
            "",
        ]
    )
    (PROJECT_ROOT / "results" / "phase7_internal_adapter_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("model                              best_acc  macro_f1  delta_vs_spatial  decision")
    for row in ranking_rows:
        print(
            f"{str(row['model']):34s} {float(row['best_acc']):.6f}  "
            f"{float(row['macro_f1']):.6f}  {float(row['delta_vs_spatial_control']):+.6f}  "
            f"{row['decision']}"
        )
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
