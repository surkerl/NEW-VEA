# NEW-VEA

PyTorch baseline project for EmotionROI using an OpenCLIP visual encoder and a linear classification head.

## Protocol

EmotionROI experiments follow the official train/test protocol. No validation split is used. Best checkpoints and early stopping are selected by official test accuracy, following the common protocol used in prior EmotionROI experiments.

This project does not create a random train/validation split and does not use `val_root`, `val_ratio`, or `val_split` configuration fields. Each epoch trains on the official train split and evaluates on the official test split. The best checkpoint is selected by `test_acc`, and early stopping monitors `test_acc` with default patience 25.

## Environment

Open PowerShell in:

```powershell
cd D:\OneDrive\Desktop\TEP-VEA\New-VEA
conda activate VEA
```

Check the environment first:

```powershell
python scripts/check_env.py
```

If packages are missing, install only the missing packages:

```powershell
python scripts/check_env.py --install-missing
```

The checker verifies:

- `torch` import
- `torch.cuda.is_available()`
- `open_clip` import
- the remaining packages from `requirements.txt`

It does not reinstall `torch` when the active `VEA` environment already has a usable torch installation.

## Dataset Inspection

The default dataset root is:

```text
D:/OneDrive/Desktop/TEP-VEA/EmotionROI
```

Inspect the official split:

```powershell
python scripts/inspect_dataset.py --root "D:\OneDrive\Desktop\TEP-VEA\EmotionROI"
```

The inspector recursively searches for official train/test split files and common train/test folder layouts. If it cannot reliably identify the official split, it stops and prints the discovered directory structure plus candidate split files. It never creates a random split.

## Smoke Test

Run a one-epoch small-sample test before full training:

```powershell
python train.py --config configs/roi_clip_baseline.yaml --epochs 1 --max_train_samples 64 --max_test_samples 64 --num_workers 0 --run_name smoke_roi_clip
```

Expected outputs:

- `logs/smoke_roi_clip.log`
- `results/smoke_roi_clip/metrics.csv`
- `checkpoints/smoke_roi_clip/best.pt`
- `checkpoints/smoke_roi_clip/last.pt`

Epoch log lines use:

```text
Epoch 003/100 | train_loss=0.8123 train_acc=0.7210 | test_loss=0.7345 test_acc=0.7562 | lr=1.00e-03 | patience=0/25 (new best)
```

The `(new best)` suffix appears only when official `test_acc` improves.

## Full Training

```powershell
python train.py --config configs/roi_clip_baseline.yaml
```

When `freeze_clip: true`, checkpoints save only the trainable classification head plus metadata. This avoids writing the full frozen OpenCLIP backbone and text tower on every epoch. The project is currently under OneDrive, so checkpoint/log/result writes can still be slowed by sync; for long runs, prefer putting run outputs on a non-OneDrive local disk if you add an output-directory override later.

Or run the PowerShell helper:

```powershell
.\scripts\run_roi_clip_baseline.ps1
```

The helper activates `VEA`, checks/install missing packages, inspects EmotionROI, trains, then evaluates the best checkpoint.

## Evaluation

```powershell
python evaluate.py --config configs/roi_clip_baseline.yaml --ckpt checkpoints/roi_clip_baseline_seed42/best.pt
```

Evaluation writes:

```text
results/roi_clip_baseline_seed42/eval_report.txt
```

## Phase 3: Spectral Evidence Diagnosis

Phase 3 does not train new models. It evaluates whether spectral presentation evidence provides a distinct affective signal beyond semantic visual representations.

Phase 3A analyzes prediction-level error differences across the Phase 2.5 fairness runs. It aligns `predictions.csv` by image path and reports per-class changes, corrected baseline errors, newly introduced errors, McNemar tests, and confusion matrices.

Phase 3B analyzes spectral perturbation robustness on the official EmotionROI test split. It probes low-pass, high-pass, band-pass, blur, downsample, amplitude-noise, and phase-noise variants, then records coarse-to-fine spectral evidence accumulation from `low_0.15` to `full`.

The diagnostic target is:

- spectral evidence differs from semantic evidence;
- affective evidence emerges across frequency bands;
- spectrum-guided models use spectral presentation cues.

Run the complete diagnosis:

```powershell
.\scripts\run_roi_phase3_spectral_diagnostics.ps1
```

Main outputs:

```text
results/phase3a_prediction_analysis/
results/phase3b_spectral_probe/
results/phase3_spectral_diagnostics_summary.md
```

## Key Defaults

- CLIP backbone: `ViT-B-16`
- pretrained weights: `openai`
- input size: `224x224`
- default mode: frozen CLIP backbone, train linear head only
- head learning rate: `1.0e-3`
- backbone learning rate when unfrozen: `1.0e-5`
- optimizer: `AdamW`
- scheduler: `CosineAnnealingLR`
- AMP: enabled on CUDA, disabled on CPU
- seed: `42`
- early stopping patience: `25`
