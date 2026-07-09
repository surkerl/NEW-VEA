import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


IMPORT_NAMES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "open_clip_torch": "open_clip",
    "timm": "timm",
    "numpy": "numpy",
    "pandas": "pandas",
    "Pillow": "PIL",
    "scikit-learn": "sklearn",
    "tqdm": "tqdm",
    "PyYAML": "yaml",
    "matplotlib": "matplotlib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and optionally install VEA Python requirements.")
    parser.add_argument("--requirements", default="requirements.txt")
    parser.add_argument("--install-missing", action="store_true")
    return parser.parse_args()


def read_requirements(path: Path) -> list[str]:
    packages: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            packages.append(line)
    return packages


def is_available(package: str) -> bool:
    import_name = IMPORT_NAMES.get(package, package)
    return importlib.util.find_spec(import_name) is not None


def main() -> int:
    args = parse_args()
    requirements_path = Path(args.requirements)
    packages = read_requirements(requirements_path)

    torch_available = is_available("torch")
    if torch_available:
        import torch

        print(f"torch: OK ({torch.__version__})")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
    else:
        print("torch: missing")

    missing = [package for package in packages if not is_available(package)]
    if torch_available:
        missing = [package for package in missing if package != "torch"]

    for package in packages:
        if package == "torch":
            continue
        print(f"{package}: {'OK' if is_available(package) else 'missing'}")

    if not missing:
        print("Environment check passed. No missing packages.")
        return 0

    print("Missing packages:")
    for package in missing:
        print(f"  - {package}")

    if not args.install_missing:
        print("Run with --install-missing to install only the missing packages.")
        return 1

    print("Installing missing packages into the active Python environment...")
    cmd = [sys.executable, "-m", "pip", "install", *missing]
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
