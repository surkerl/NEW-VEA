$ErrorActionPreference = "Stop"

conda activate VEA
if ($LASTEXITCODE -ne 0) {
    throw "Failed to activate conda environment VEA."
}

python scripts/check_env.py --install-missing
if ($LASTEXITCODE -ne 0) {
    throw "Environment check failed."
}

python scripts/inspect_dataset.py --root "D:\OneDrive\Desktop\TEP-VEA\EmotionROI"
if ($LASTEXITCODE -ne 0) {
    throw "Dataset inspection failed."
}

python train.py --config configs/roi_clip_baseline.yaml
if ($LASTEXITCODE -ne 0) {
    throw "Training failed."
}

python evaluate.py --config configs/roi_clip_baseline.yaml --ckpt checkpoints/roi_clip_baseline_seed42/best.pt
if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed."
}
