import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_ALIASES = {"train", "training"}
TEST_ALIASES = {"test", "testing"}
IMAGE_ROOT_CANDIDATES = ("images", "Images", "image", "Image", "img", "imgs", "data")


class DatasetSplitError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawSample:
    path: Path
    label: str
    source: str


@dataclass(frozen=True)
class IndexedSample:
    path: str
    label: int


@dataclass(frozen=True)
class SplitDiscovery:
    split_type: str
    train: list[IndexedSample]
    test: list[IndexedSample]
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    split_files: dict[str, str]


def build_clip_transform(split: str, input_size: int = 224) -> transforms.Compose:
    split = _normalize_split_name(split)
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    input_size,
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


class EmotionROIDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        input_size: int = 224,
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
        discovery: SplitDiscovery | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = _normalize_split_name(split)
        self.discovery = discovery or discover_emotionroi_splits(self.root)
        self.class_to_idx = dict(self.discovery.class_to_idx)
        self.idx_to_class = dict(self.discovery.idx_to_class)
        self.transform = transform or build_clip_transform(self.split, input_size)

        samples = self.discovery.train if self.split == "train" else self.discovery.test
        if max_samples is not None:
            samples = samples[: max(0, int(max_samples))]
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
        return {
            "image": self.transform(image),
            "label": int(sample.label),
            "path": str(sample.path),
        }


def discover_emotionroi_splits(root: str | Path) -> SplitDiscovery:
    root = Path(root)
    if not root.exists():
        raise DatasetSplitError(f"Dataset root does not exist: {root}")

    split_candidates = _find_split_file_candidates(root)
    errors: list[str] = []
    for train_file, test_file in _rank_split_pairs(split_candidates):
        try:
            train_raw = _parse_split_file(train_file, root)
            test_raw = _parse_split_file(test_file, root)
            train, test, class_to_idx, idx_to_class = _index_samples(train_raw, test_raw)
            return SplitDiscovery(
                split_type="split_files",
                train=train,
                test=test,
                class_to_idx=class_to_idx,
                idx_to_class=idx_to_class,
                split_files={"train": str(train_file), "test": str(test_file)},
            )
        except Exception as exc:
            errors.append(f"{train_file} / {test_file}: {exc}")

    folder_discovery = _discover_folder_split(root)
    if folder_discovery is not None:
        return folder_discovery

    message = [
        "Could not reliably identify the official EmotionROI train/test split.",
        "No random train/val split was created.",
        "",
        "Directory structure:",
        describe_directory_structure(root),
        "",
        "Candidate split files:",
    ]
    if split_candidates:
        message.extend(f"- {path}" for _, path in split_candidates)
    else:
        message.append("- none")
    if errors:
        message.extend(["", "Candidate parsing errors:"])
        message.extend(f"- {item}" for item in errors[:10])
    raise DatasetSplitError("\n".join(message))


def describe_directory_structure(root: str | Path, max_depth: int = 3, max_entries: int = 80) -> str:
    root = Path(root)
    lines: list[str] = []
    count = 0

    def walk(path: Path, depth: int) -> None:
        nonlocal count
        if count >= max_entries:
            return
        indent = "  " * depth
        label = path.name + ("/" if path.is_dir() else "")
        lines.append(f"{indent}{label}")
        count += 1
        if depth >= max_depth or not path.is_dir():
            return
        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for child in children:
            if count >= max_entries:
                lines.append(f"{indent}  ...")
                count += 1
                return
            walk(child, depth + 1)

    walk(root, 0)
    return "\n".join(lines)


def _normalize_split_name(split: str) -> str:
    value = split.lower()
    if value in TRAIN_ALIASES:
        return "train"
    if value in TEST_ALIASES:
        return "test"
    raise ValueError(f"Unsupported split '{split}'. EmotionROI supports only train and test.")


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _find_split_file_candidates(root: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".csv", ".tsv"}:
            continue
        role = _split_file_role(path)
        if role is not None:
            candidates.append((role, path))
    return candidates


def _split_file_role(path: Path) -> str | None:
    stem = path.stem.lower()
    if stem in TRAIN_ALIASES or "train" in stem:
        return "train"
    if stem in TEST_ALIASES or "test" in stem:
        return "test"
    return None


def _rank_split_pairs(candidates: list[tuple[str, Path]]) -> list[tuple[Path, Path]]:
    train_files = [path for role, path in candidates if role == "train"]
    test_files = [path for role, path in candidates if role == "test"]
    pairs = [(train_file, test_file) for train_file in train_files for test_file in test_files]

    def score(pair: tuple[Path, Path]) -> tuple[int, str]:
        train_file, test_file = pair
        value = 0
        if train_file.parent == test_file.parent:
            value += 100
        parent_text = "/".join(part.lower() for part in train_file.parent.parts)
        if "split" in parent_text:
            value += 30
        if "training_testing" in parent_text or "train_test" in parent_text:
            value += 40
        if train_file.stem.lower() in TRAIN_ALIASES:
            value += 10
        if test_file.stem.lower() in TEST_ALIASES:
            value += 10
        return value, str(train_file)

    return sorted(pairs, key=score, reverse=True)


def _parse_split_file(split_file: Path, root: Path) -> list[RawSample]:
    samples: list[RawSample] = []
    header: dict[str, int | None] | None = None

    with open(split_file, "r", encoding="utf-8-sig", newline="") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = _split_line(line)
            if not parts:
                continue

            if header is None and _looks_like_header(parts):
                header = _header_indices(parts)
                continue

            try:
                raw_path, raw_label = _extract_path_label(parts, header)
                resolved_path = _resolve_image_path(raw_path, root, split_file)
                label = raw_label if raw_label is not None else _infer_label(raw_path, resolved_path, root)
            except Exception as exc:
                raise DatasetSplitError(f"{split_file}:{line_number}: {exc}") from exc

            if label is None or str(label).strip() == "":
                raise DatasetSplitError(f"{split_file}:{line_number}: missing label")
            samples.append(RawSample(path=resolved_path, label=str(label).strip(), source=raw_path))

    if not samples:
        raise DatasetSplitError(f"No samples parsed from {split_file}")
    return samples


def _split_line(line: str) -> list[str]:
    if "," in line:
        return [part.strip().strip('"').strip("'") for part in next(csv.reader([line]))]
    if "\t" in line:
        return [part.strip().strip('"').strip("'") for part in line.split("\t")]
    return [part.strip().strip('"').strip("'") for part in line.split()]


def _looks_like_header(parts: list[str]) -> bool:
    lowered = [part.lower().strip() for part in parts]
    path_names = {"image", "image_path", "img", "path", "filepath", "file", "filename", "image_name"}
    label_names = {"label", "class", "class_name", "emotion", "category", "target"}
    return any(part in path_names for part in lowered) and (
        len(parts) == 1 or any(part in label_names for part in lowered)
    )


def _header_indices(parts: list[str]) -> dict[str, int | None]:
    lowered = [part.lower().strip() for part in parts]
    path_names = {"image", "image_path", "img", "path", "filepath", "file", "filename", "image_name"}
    label_names = {"label", "class", "class_name", "emotion", "category", "target"}
    path_idx = next((i for i, part in enumerate(lowered) if part in path_names), 0)
    label_idx = next((i for i, part in enumerate(lowered) if part in label_names), None)
    return {"path": path_idx, "label": label_idx}


def _extract_path_label(parts: list[str], header: dict[str, int | None] | None) -> tuple[str, str | None]:
    if header is not None:
        path_idx = int(header["path"] or 0)
        label_idx = header["label"]
        raw_path = parts[path_idx]
        raw_label = parts[int(label_idx)] if label_idx is not None and int(label_idx) < len(parts) else None
        return raw_path, raw_label

    raw_path = parts[0]
    raw_label = parts[1] if len(parts) > 1 else None
    return raw_path, raw_label


def _resolve_image_path(raw_path: str, root: Path, split_file: Path) -> Path:
    raw_clean = raw_path.strip().strip('"').strip("'")
    path_obj = Path(raw_clean)
    if path_obj.is_absolute() and _is_image_file(path_obj):
        return path_obj

    rel = Path(raw_clean.replace("\\", "/"))
    candidates = [root / rel]
    candidates.extend(root / image_root / rel for image_root in IMAGE_ROOT_CANDIDATES)
    candidates.append(split_file.parent / rel)

    for candidate in candidates:
        if _is_image_file(candidate):
            return candidate

    raise FileNotFoundError(f"image path not found: {raw_path}")


def _infer_label(raw_path: str, resolved_path: Path, root: Path) -> str | None:
    normalized = raw_path.replace("\\", "/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in {"", "."}]
    if len(parts) >= 2:
        first = parts[0]
        if first in IMAGE_ROOT_CANDIDATES and len(parts) >= 3:
            return parts[1]
        return first

    for image_root in IMAGE_ROOT_CANDIDATES:
        try:
            rel = resolved_path.relative_to(root / image_root)
            if len(rel.parts) >= 2:
                return rel.parts[0]
        except ValueError:
            pass

    if resolved_path.parent != root:
        return resolved_path.parent.name
    return None


def _index_samples(
    train_raw: list[RawSample],
    test_raw: list[RawSample],
) -> tuple[list[IndexedSample], list[IndexedSample], dict[str, int], dict[int, str]]:
    all_labels = [sample.label for sample in train_raw + test_raw]
    numeric = all(_is_int_label(label) for label in all_labels)

    if numeric:
        values = sorted({int(label) for label in all_labels})
        if values == list(range(len(values))):
            class_to_idx = {str(value): value for value in values}
        else:
            class_to_idx = {str(value): index for index, value in enumerate(values)}
        idx_to_class = {index: label for label, index in class_to_idx.items()}
    else:
        labels = sorted(set(all_labels))
        class_to_idx = {label: index for index, label in enumerate(labels)}
        idx_to_class = {index: label for label, index in class_to_idx.items()}

    def convert(samples: Iterable[RawSample]) -> list[IndexedSample]:
        converted = []
        for sample in samples:
            key = str(int(sample.label)) if numeric else sample.label
            converted.append(IndexedSample(path=str(sample.path), label=class_to_idx[key]))
        return converted

    return convert(train_raw), convert(test_raw), class_to_idx, idx_to_class


def _is_int_label(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def _discover_folder_split(root: Path) -> SplitDiscovery | None:
    dirs = {child.name.lower(): child for child in root.iterdir() if child.is_dir()}
    for train_name in TRAIN_ALIASES:
        for test_name in TEST_ALIASES:
            train_dir = dirs.get(train_name)
            test_dir = dirs.get(test_name)
            if train_dir is None or test_dir is None:
                continue
            train_raw, test_raw = _read_class_folders(train_dir, test_dir)
            if train_raw and test_raw:
                train, test, class_to_idx, idx_to_class = _index_samples(train_raw, test_raw)
                return SplitDiscovery(
                    split_type="folder_split",
                    train=train,
                    test=test,
                    class_to_idx=class_to_idx,
                    idx_to_class=idx_to_class,
                    split_files={},
                )
    return None


def _read_class_folders(train_dir: Path, test_dir: Path) -> tuple[list[RawSample], list[RawSample]]:
    class_names = sorted(
        {
            child.name
            for parent in (train_dir, test_dir)
            for child in parent.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        }
    )

    def collect(parent: Path) -> list[RawSample]:
        samples: list[RawSample] = []
        for class_name in class_names:
            class_dir = parent / class_name
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.rglob("*"), key=lambda p: str(p).lower()):
                if _is_image_file(image_path):
                    samples.append(RawSample(path=image_path, label=class_name, source=str(image_path)))
        return samples

    return collect(train_dir), collect(test_dir)
