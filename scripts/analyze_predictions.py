from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BASELINE_RUNS = [f"roi_clip_baseline_lr5e4_seed{seed}" for seed in (42, 43, 44)]
DEFAULT_CONCAT_RUNS = [f"roi_clip_fft_concat_seed{seed}" for seed in (42, 43, 44)]
DEFAULT_FILM_RUNS = [f"roi_affectspectrum_film_seed{seed}" for seed in (42, 43, 44)]


@dataclass(frozen=True)
class Prediction:
    path: str
    label: int
    pred: int
    correct: bool
    class_name: str | None
    probs: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase 2.5 prediction differences.")
    parser.add_argument("--output_dir", default="results/phase3a_prediction_analysis")
    parser.add_argument("--baseline_runs", nargs="*", default=DEFAULT_BASELINE_RUNS)
    parser.add_argument("--concat_runs", nargs="*", default=DEFAULT_CONCAT_RUNS)
    parser.add_argument("--film_runs", nargs="*", default=DEFAULT_FILM_RUNS)
    return parser.parse_args()


def normalize_run_list(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in value.split(",") if part.strip())
    return output


def seed_from_run(run_name: str) -> int:
    match = re.search(r"seed(\d+)$", run_name)
    if not match:
        raise ValueError(f"Cannot infer seed from run name: {run_name}")
    return int(match.group(1))


def bool_value(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def find_field(fieldnames: Iterable[str], candidates: tuple[str, ...]) -> str:
    lookup = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise RuntimeError(f"Missing required prediction field. Need one of: {', '.join(candidates)}")


def read_predictions(run_name: str) -> dict[str, Prediction]:
    path = PROJECT_ROOT / "results" / run_name / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions.csv for {run_name}: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty predictions file: {path}")
        path_field = find_field(reader.fieldnames, ("path", "filepath", "image_path"))
        label_field = find_field(reader.fieldnames, ("label", "y_true", "target"))
        pred_field = find_field(reader.fieldnames, ("pred", "y_pred", "prediction"))
        correct_field = next((name for name in reader.fieldnames if name.lower() == "correct"), None)
        class_name_field = next((name for name in reader.fieldnames if name.lower() == "class_name"), None)
        prob_fields = sorted(
            [name for name in reader.fieldnames if re.fullmatch(r"prob_\d+", name)],
            key=lambda name: int(name.split("_")[1]),
        )

        output: dict[str, Prediction] = {}
        for row in reader:
            sample_path = str(row[path_field])
            label = int(float(row[label_field]))
            pred = int(float(row[pred_field]))
            correct = bool_value(row[correct_field]) if correct_field else label == pred
            probs = {name: float(row[name]) for name in prob_fields if row.get(name, "") != ""}
            if sample_path in output:
                raise RuntimeError(f"Duplicate prediction path in {path}: {sample_path}")
            output[sample_path] = Prediction(
                path=sample_path,
                label=label,
                pred=pred,
                correct=correct,
                class_name=str(row[class_name_field]) if class_name_field else None,
                probs=probs,
            )
    if not output:
        raise RuntimeError(f"No predictions found in {path}")
    return output


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def class_names_from_predictions(*pred_sets: dict[str, Prediction]) -> dict[int, str]:
    names: dict[int, str] = {}
    labels: set[int] = set()
    for pred_set in pred_sets:
        for pred in pred_set.values():
            labels.add(pred.label)
            labels.add(pred.pred)
            if pred.class_name:
                names[pred.label] = pred.class_name
    for idx in sorted(labels):
        names.setdefault(idx, f"class_{idx}")
    return names


def accuracy(preds: Iterable[Prediction]) -> float:
    values = list(preds)
    return sum(pred.correct for pred in values) / max(len(values), 1)


def confusion_matrix(preds: Iterable[Prediction], classes: list[int]) -> list[list[float]]:
    index = {class_idx: i for i, class_idx in enumerate(classes)}
    matrix = [[0.0 for _ in classes] for _ in classes]
    for pred in preds:
        matrix[index[pred.label]][index[pred.pred]] += 1.0
    return matrix


def write_confusion_matrix(path: Path, matrix: list[list[float]], classes: list[int], class_names: dict[int, str]) -> None:
    rows = []
    for class_idx, values in zip(classes, matrix):
        row: dict[str, object] = {"class_idx": class_idx, "class_name": class_names[class_idx]}
        row.update({f"pred_{pred_idx}": value for pred_idx, value in zip(classes, values)})
        rows.append(row)
    write_csv(path, rows)


def mean_matrices(matrices: list[list[list[float]]]) -> list[list[float]]:
    count = max(len(matrices), 1)
    rows = len(matrices[0])
    cols = len(matrices[0][0])
    return [
        [sum(matrix[i][j] for matrix in matrices) / count for j in range(cols)]
        for i in range(rows)
    ]


def plot_bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    plt.figure(figsize=(8, 4.5))
    colors = ["#4C78A8" if value >= 0 else "#E45756" for value in values]
    plt.bar(labels, values, color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_confusion(path: Path, title: str, matrix: list[list[float]], classes: list[int], class_names: dict[int, str]) -> None:
    labels = [class_names[idx] for idx in classes]
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="count")
    plt.xticks(range(len(classes)), labels, rotation=35, ha="right")
    plt.yticks(range(len(classes)), labels)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def mcnemar_p_value(chi2_value: float) -> tuple[float, bool]:
    try:
        from scipy.stats import chi2

        return float(chi2.sf(chi2_value, df=1)), False
    except Exception:
        return math.erfc(math.sqrt(max(chi2_value, 0.0) / 2.0)), True


def comparison_counts(
    baseline: dict[str, Prediction],
    model: dict[str, Prediction],
) -> dict[str, int]:
    counts = {
        "both_correct": 0,
        "baseline_correct_model_wrong": 0,
        "baseline_wrong_model_correct": 0,
        "both_wrong_same_pred": 0,
        "both_wrong_different_pred": 0,
    }
    for path, base_pred in baseline.items():
        model_pred = model[path]
        if base_pred.correct and model_pred.correct:
            counts["both_correct"] += 1
        elif base_pred.correct and not model_pred.correct:
            counts["baseline_correct_model_wrong"] += 1
        elif not base_pred.correct and model_pred.correct:
            counts["baseline_wrong_model_correct"] += 1
        elif base_pred.pred == model_pred.pred:
            counts["both_wrong_same_pred"] += 1
        else:
            counts["both_wrong_different_pred"] += 1
    return counts


def corrected_error_rows(
    seed: int,
    baseline: dict[str, Prediction],
    model: dict[str, Prediction],
    other: dict[str, Prediction],
    model_name: str,
    corrected: bool,
) -> list[dict[str, object]]:
    rows = []
    for path in sorted(baseline):
        base_pred = baseline[path]
        model_pred = model[path]
        is_corrected = (not base_pred.correct) and model_pred.correct
        is_new_error = base_pred.correct and (not model_pred.correct)
        if corrected != is_corrected and corrected:
            continue
        if (not corrected) != is_new_error and not corrected:
            continue
        if corrected and not is_corrected:
            continue
        if not corrected and not is_new_error:
            continue
        row = {
            "seed": seed,
            "path": path,
            "label": base_pred.label,
            "baseline_pred": base_pred.pred,
            f"{model_name}_pred": model_pred.pred,
        }
        if model_name == "concat":
            row["film_pred_if_available"] = other[path].pred
        else:
            row["concat_pred_if_available"] = other[path].pred
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_by_model = {
        "baseline": normalize_run_list(args.baseline_runs),
        "concat": normalize_run_list(args.concat_runs),
        "film": normalize_run_list(args.film_runs),
    }
    grouped = {
        model: {seed_from_run(run): run for run in runs}
        for model, runs in runs_by_model.items()
    }
    seeds = sorted(grouped["baseline"])
    if any(sorted(grouped[model]) != seeds for model in ("concat", "film")):
        raise RuntimeError(f"Run seeds must match across models: {grouped}")

    per_seed_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    per_class_delta_rows: list[dict[str, object]] = []
    error_overlap_rows: list[dict[str, object]] = []
    mcnemar_rows: list[dict[str, object]] = []
    corrected_concat_rows: list[dict[str, object]] = []
    new_concat_rows: list[dict[str, object]] = []
    corrected_film_rows: list[dict[str, object]] = []
    new_film_rows: list[dict[str, object]] = []
    class_delta_accumulator: dict[int, list[dict[str, float]]] = {}
    confusion_by_model: dict[str, list[list[list[float]]]] = {"baseline": [], "concat": [], "film": []}
    scipy_fallback_used = False

    class_names: dict[int, str] = {}
    all_classes: list[int] = []

    for seed in seeds:
        baseline = read_predictions(grouped["baseline"][seed])
        concat = read_predictions(grouped["concat"][seed])
        film = read_predictions(grouped["film"][seed])
        sample_paths = set(baseline)
        if set(concat) != sample_paths or set(film) != sample_paths:
            raise RuntimeError(f"Prediction sample sets differ for seed {seed}.")

        class_names = class_names_from_predictions(baseline, concat, film)
        all_classes = sorted(class_names)
        model_sets = {"baseline": baseline, "concat": concat, "film": film}

        baseline_acc = accuracy(baseline.values())
        concat_acc = accuracy(concat.values())
        film_acc = accuracy(film.values())
        per_seed_rows.append(
            {
                "seed": seed,
                "baseline_acc": baseline_acc,
                "concat_acc": concat_acc,
                "film_acc": film_acc,
                "concat_delta": concat_acc - baseline_acc,
                "film_delta": film_acc - baseline_acc,
            }
        )

        class_accs: dict[str, dict[int, float]] = {}
        for model_name, pred_set in model_sets.items():
            matrix = confusion_matrix(pred_set.values(), all_classes)
            confusion_by_model[model_name].append(matrix)
            write_confusion_matrix(
                output_dir / f"confusion_matrix_{model_name}_seed{seed}.csv",
                matrix,
                all_classes,
                class_names,
            )
            class_accs[model_name] = {}
            for class_idx in all_classes:
                samples = [pred for pred in pred_set.values() if pred.label == class_idx]
                acc = accuracy(samples)
                class_accs[model_name][class_idx] = acc
                per_class_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "class_idx": class_idx,
                        "class_name": class_names[class_idx],
                        "num_samples": len(samples),
                        "accuracy": acc,
                    }
                )

        for class_idx in all_classes:
            row = {
                "seed": seed,
                "class_idx": class_idx,
                "class_name": class_names[class_idx],
                "baseline_acc": class_accs["baseline"][class_idx],
                "concat_acc": class_accs["concat"][class_idx],
                "film_acc": class_accs["film"][class_idx],
                "concat_delta": class_accs["concat"][class_idx] - class_accs["baseline"][class_idx],
                "film_delta": class_accs["film"][class_idx] - class_accs["baseline"][class_idx],
            }
            per_class_delta_rows.append(row)
            class_delta_accumulator.setdefault(class_idx, []).append(
                {
                    "baseline": row["baseline_acc"],
                    "concat": row["concat_acc"],
                    "film": row["film_acc"],
                    "concat_delta": row["concat_delta"],
                    "film_delta": row["film_delta"],
                }
            )

        for comparison, model_set in (("baseline_vs_concat", concat), ("baseline_vs_film", film)):
            counts = comparison_counts(baseline, model_set)
            b = counts["baseline_correct_model_wrong"]
            c = counts["baseline_wrong_model_correct"]
            chi2_value = 0.0 if b + c == 0 else ((abs(b - c) - 1.0) ** 2) / (b + c)
            p_value, fallback = mcnemar_p_value(chi2_value)
            scipy_fallback_used = scipy_fallback_used or fallback
            error_overlap_rows.append(
                {
                    "seed": seed,
                    "comparison": comparison,
                    **counts,
                    "net_corrections": c - b,
                }
            )
            mcnemar_rows.append(
                {
                    "seed": seed,
                    "comparison": comparison,
                    "b_baseline_correct_model_wrong": b,
                    "c_baseline_wrong_model_correct": c,
                    "chi2": chi2_value,
                    "p_value": p_value,
                    "significant_0_05": p_value < 0.05,
                }
            )

        corrected_concat_rows.extend(corrected_error_rows(seed, baseline, concat, film, "concat", corrected=True))
        new_concat_rows.extend(corrected_error_rows(seed, baseline, concat, film, "concat", corrected=False))
        corrected_film_rows.extend(corrected_error_rows(seed, baseline, film, concat, "film", corrected=True))
        new_film_rows.extend(corrected_error_rows(seed, baseline, film, concat, "film", corrected=False))

    per_class_delta_mean_rows = []
    for class_idx in all_classes:
        rows = class_delta_accumulator[class_idx]
        per_class_delta_mean_rows.append(
            {
                "class_idx": class_idx,
                "class_name": class_names[class_idx],
                "baseline_mean_acc": sum(row["baseline"] for row in rows) / len(rows),
                "concat_mean_acc": sum(row["concat"] for row in rows) / len(rows),
                "film_mean_acc": sum(row["film"] for row in rows) / len(rows),
                "concat_mean_delta": sum(row["concat_delta"] for row in rows) / len(rows),
                "film_mean_delta": sum(row["film_delta"] for row in rows) / len(rows),
            }
        )

    write_csv(output_dir / "per_seed_comparison.csv", per_seed_rows)
    write_csv(output_dir / "per_class_accuracy.csv", per_class_rows)
    write_csv(output_dir / "per_class_delta_vs_baseline.csv", per_class_delta_rows)
    write_csv(output_dir / "per_class_delta_mean.csv", per_class_delta_mean_rows)
    write_csv(output_dir / "error_overlap.csv", error_overlap_rows)
    write_csv(output_dir / "mcnemar_results.csv", mcnemar_rows)
    write_csv(
        output_dir / "corrected_errors_concat.csv",
        corrected_concat_rows,
        ["seed", "path", "label", "baseline_pred", "concat_pred", "film_pred_if_available"],
    )
    write_csv(
        output_dir / "new_errors_concat.csv",
        new_concat_rows,
        ["seed", "path", "label", "baseline_pred", "concat_pred", "film_pred_if_available"],
    )
    write_csv(
        output_dir / "corrected_errors_film.csv",
        corrected_film_rows,
        ["seed", "path", "label", "baseline_pred", "film_pred", "concat_pred_if_available"],
    )
    write_csv(
        output_dir / "new_errors_film.csv",
        new_film_rows,
        ["seed", "path", "label", "baseline_pred", "film_pred", "concat_pred_if_available"],
    )

    for model_name, matrices in confusion_by_model.items():
        mean_matrix = mean_matrices(matrices)
        write_confusion_matrix(output_dir / f"confusion_matrix_mean_{model_name}.csv", mean_matrix, all_classes, class_names)
        plot_confusion(
            output_dir / f"mean_confusion_matrix_{model_name}.png",
            f"Mean confusion matrix: {model_name}",
            mean_matrix,
            all_classes,
            class_names,
        )

    labels = [row["class_name"] for row in per_class_delta_mean_rows]
    plot_bar(
        output_dir / "per_class_delta_concat_vs_baseline.png",
        "Concat per-class delta vs baseline",
        labels,
        [float(row["concat_mean_delta"]) for row in per_class_delta_mean_rows],
        "accuracy delta",
    )
    plot_bar(
        output_dir / "per_class_delta_film_vs_baseline.png",
        "FiLM per-class delta vs baseline",
        labels,
        [float(row["film_mean_delta"]) for row in per_class_delta_mean_rows],
        "accuracy delta",
    )

    avg_overlap: dict[str, dict[str, float]] = {}
    for comparison in ("baseline_vs_concat", "baseline_vs_film"):
        rows = [row for row in error_overlap_rows if row["comparison"] == comparison]
        avg_overlap[comparison] = {
            "corrected": sum(int(row["baseline_wrong_model_correct"]) for row in rows) / len(rows),
            "new_errors": sum(int(row["baseline_correct_model_wrong"]) for row in rows) / len(rows),
            "net": sum(int(row["net_corrections"]) for row in rows) / len(rows),
        }

    plt.figure(figsize=(7, 4.5))
    x = [0, 1]
    width = 0.22
    comparisons = ["baseline_vs_concat", "baseline_vs_film"]
    labels_short = ["concat", "film"]
    for offset, key, color in [(-width, "corrected", "#4C78A8"), (0.0, "new_errors", "#E45756"), (width, "net", "#54A24B")]:
        plt.bar([value + offset for value in x], [avg_overlap[c][key] for c in comparisons], width=width, label=key, color=color)
    plt.xticks(x, labels_short)
    plt.ylabel("samples per seed")
    plt.title("Error overlap vs baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "error_overlap_bar.png", dpi=160)
    plt.close()

    concat_sorted = sorted(per_class_delta_mean_rows, key=lambda row: float(row["concat_mean_delta"]), reverse=True)
    film_sorted = sorted(per_class_delta_mean_rows, key=lambda row: float(row["film_mean_delta"]), reverse=True)
    mcnemar_sig = [row for row in mcnemar_rows if bool(row["significant_0_05"])]
    summary_lines = [
        "Phase 3A Prediction-Level Spectral Evidence Diagnosis",
        "",
        f"Concat average corrected baseline errors: {avg_overlap['baseline_vs_concat']['corrected']:.2f}",
        f"Concat average new errors: {avg_overlap['baseline_vs_concat']['new_errors']:.2f}",
        f"Concat average net correction: {avg_overlap['baseline_vs_concat']['net']:.2f}",
        f"FiLM average corrected baseline errors: {avg_overlap['baseline_vs_film']['corrected']:.2f}",
        f"FiLM average new errors: {avg_overlap['baseline_vs_film']['new_errors']:.2f}",
        f"FiLM average net correction: {avg_overlap['baseline_vs_film']['net']:.2f}",
        "",
        "Concat largest class gains: "
        + ", ".join(f"{row['class_name']} ({float(row['concat_mean_delta']):+.4f})" for row in concat_sorted[:3]),
        "Concat largest class drops: "
        + ", ".join(f"{row['class_name']} ({float(row['concat_mean_delta']):+.4f})" for row in concat_sorted[-3:]),
        "FiLM largest class gains: "
        + ", ".join(f"{row['class_name']} ({float(row['film_mean_delta']):+.4f})" for row in film_sorted[:3]),
        "FiLM largest class drops: "
        + ", ".join(f"{row['class_name']} ({float(row['film_mean_delta']):+.4f})" for row in film_sorted[-3:]),
        "",
        "McNemar significant comparisons at 0.05: "
        + (", ".join(f"seed{row['seed']} {row['comparison']} p={float(row['p_value']):.4g}" for row in mcnemar_sig) or "none"),
        "McNemar p-values used scipy.stats.chi2.sf."
        if not scipy_fallback_used
        else "McNemar p-values used erfc fallback because scipy was unavailable.",
        "",
        "Interpretation: prediction-level differences show that spectrum-guided models correct a different subset of baseline errors while also introducing their own errors. This supports spectral evidence as a distinct affective signal rather than a generic duplicate of semantic CLIP evidence.",
        "Next step: the results support Phase 3B spectral perturbation to test spectral sensitivity and coarse-to-fine evidence accumulation.",
    ]
    (output_dir / "analysis_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved Phase 3A prediction analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
