import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.emotionroi import DatasetSplitError, discover_emotionroi_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect EmotionROI official train/test split.")
    parser.add_argument("--root", required=True, help="EmotionROI dataset root.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    print(f"Dataset root: {root}")

    try:
        discovery = discover_emotionroi_splits(root)
    except DatasetSplitError as exc:
        print(str(exc))
        return 1

    print(f"Detected split form: {discovery.split_type}")
    if discovery.split_files:
        print(f"Train split file: {discovery.split_files['train']}")
        print(f"Test split file: {discovery.split_files['test']}")
    print(f"Train samples: {len(discovery.train)}")
    print(f"Test samples: {len(discovery.test)}")
    print(f"Num classes: {len(discovery.class_to_idx)}")
    print("class_to_idx:")
    print(json.dumps(discovery.class_to_idx, indent=2, sort_keys=True))
    print("First 5 train samples:")
    for sample in discovery.train[:5]:
        print(f"  label={sample.label} path={sample.path}")
    print("First 5 test samples:")
    for sample in discovery.test[:5]:
        print(f"  label={sample.label} path={sample.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
