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

& $Python scripts/analyze_predictions.py --output_dir results/phase3a_prediction_analysis
if ($LASTEXITCODE -ne 0) { throw "Phase 3A prediction analysis failed." }

$runs = @(
    @{ Config = "configs/roi_clip_baseline.yaml"; Run = "roi_clip_baseline_lr5e4_seed42"; Seed = 42 },
    @{ Config = "configs/roi_clip_baseline.yaml"; Run = "roi_clip_baseline_lr5e4_seed43"; Seed = 43 },
    @{ Config = "configs/roi_clip_baseline.yaml"; Run = "roi_clip_baseline_lr5e4_seed44"; Seed = 44 },
    @{ Config = "configs/roi_clip_fft_concat.yaml"; Run = "roi_clip_fft_concat_seed42"; Seed = 42 },
    @{ Config = "configs/roi_clip_fft_concat.yaml"; Run = "roi_clip_fft_concat_seed43"; Seed = 43 },
    @{ Config = "configs/roi_clip_fft_concat.yaml"; Run = "roi_clip_fft_concat_seed44"; Seed = 44 },
    @{ Config = "configs/roi_affectspectrum_film.yaml"; Run = "roi_affectspectrum_film_seed42"; Seed = 42 },
    @{ Config = "configs/roi_affectspectrum_film.yaml"; Run = "roi_affectspectrum_film_seed43"; Seed = 43 },
    @{ Config = "configs/roi_affectspectrum_film.yaml"; Run = "roi_affectspectrum_film_seed44"; Seed = 44 }
)

foreach ($item in $runs) {
    $ckpt = "checkpoints/$($item.Run)/best.pt"
    & $Python scripts/spectral_probe.py `
        --config $item.Config `
        --ckpt $ckpt `
        --run_name $item.Run `
        --probe_name $item.Run `
        --output_dir results/phase3b_spectral_probe `
        --seed $item.Seed
    if ($LASTEXITCODE -ne 0) { throw "Spectral probe failed for $($item.Run)." }
}

& $Python scripts/summarize_phase3_diagnostics.py `
    --phase3a_dir results/phase3a_prediction_analysis `
    --phase3b_dir results/phase3b_spectral_probe `
    --output_dir results
if ($LASTEXITCODE -ne 0) { throw "Phase 3 diagnostics summary failed." }
