$ErrorActionPreference = "Stop"

conda activate VEA
if ($LASTEXITCODE -ne 0) {
    throw "Failed to activate conda environment VEA."
}

$Python = "python"
try {
    & $Python -c "import open_clip" *> $null
    $CanImportOpenClip = ($LASTEXITCODE -eq 0)
} catch {
    $CanImportOpenClip = $false
}
if (-not $CanImportOpenClip) {
    $CondaBase = conda info --base
    $CandidatePython = Join-Path $CondaBase "envs\VEA\python.exe"
    if (-not (Test-Path $CandidatePython)) {
        throw "Could not find VEA python at $CandidatePython."
    }
    $Python = $CandidatePython
    & $Python -c "import open_clip"
    if ($LASTEXITCODE -ne 0) {
        throw "VEA python exists but cannot import open_clip."
    }
}

& $Python -m compileall src train.py evaluate.py scripts
if ($LASTEXITCODE -ne 0) { throw "Compile check failed." }

& $Python train.py `
    --config configs/roi_affectspectrum_gated.yaml `
    --epochs 2 `
    --max_train_samples 64 `
    --max_test_samples 64 `
    --num_workers 0 `
    --run_name smoke_affectspectrum_gated
if ($LASTEXITCODE -ne 0) { throw "Smoke training failed." }

& $Python evaluate.py `
    --config configs/roi_affectspectrum_gated.yaml `
    --ckpt checkpoints/smoke_affectspectrum_gated/best.pt `
    --run_name smoke_affectspectrum_gated `
    --num_workers 0
if ($LASTEXITCODE -ne 0) { throw "Smoke evaluation failed." }

& $Python train.py `
    --config configs/roi_affectspectrum_gated.yaml `
    --seed 42 `
    --run_name roi_affectspectrum_gated_seed42
if ($LASTEXITCODE -ne 0) { throw "Seed 42 training failed." }

& $Python evaluate.py `
    --config configs/roi_affectspectrum_gated.yaml `
    --ckpt checkpoints/roi_affectspectrum_gated_seed42/best.pt `
    --run_name roi_affectspectrum_gated_seed42
if ($LASTEXITCODE -ne 0) { throw "Seed 42 evaluation failed." }

& $Python train.py `
    --config configs/roi_affectspectrum_gated.yaml `
    --seed 43 `
    --run_name roi_affectspectrum_gated_seed43
if ($LASTEXITCODE -ne 0) { throw "Seed 43 training failed." }

& $Python evaluate.py `
    --config configs/roi_affectspectrum_gated.yaml `
    --ckpt checkpoints/roi_affectspectrum_gated_seed43/best.pt `
    --run_name roi_affectspectrum_gated_seed43
if ($LASTEXITCODE -ne 0) { throw "Seed 43 evaluation failed." }

& $Python train.py `
    --config configs/roi_affectspectrum_gated.yaml `
    --seed 44 `
    --run_name roi_affectspectrum_gated_seed44
if ($LASTEXITCODE -ne 0) { throw "Seed 44 training failed." }

& $Python evaluate.py `
    --config configs/roi_affectspectrum_gated.yaml `
    --ckpt checkpoints/roi_affectspectrum_gated_seed44/best.pt `
    --run_name roi_affectspectrum_gated_seed44
if ($LASTEXITCODE -ne 0) { throw "Seed 44 evaluation failed." }

foreach ($seed in 42, 43, 44) {
    $run = "roi_affectspectrum_gated_seed$seed"
    & $Python scripts/spectral_probe.py `
        --config configs/roi_affectspectrum_gated.yaml `
        --ckpt "checkpoints/$run/best.pt" `
        --run_name $run `
        --probe_name $run `
        --output_dir results/phase4_gated_spectral_probe `
        --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Spectral probe failed for $run." }
}

& $Python scripts/summarize_phase4_gated.py
if ($LASTEXITCODE -ne 0) { throw "Phase 4 gated summary failed." }
