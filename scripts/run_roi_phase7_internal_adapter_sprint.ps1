$ErrorActionPreference = "Stop"

conda activate VEA
if ($LASTEXITCODE -ne 0) {
    throw "Failed to activate conda environment VEA."
}

$CondaBase = conda info --base
$Python = Join-Path $CondaBase "envs\VEA\python.exe"
if (-not (Test-Path $Python)) {
    throw "Could not find VEA python at $Python."
}
& $Python -c "import open_clip"
if ($LASTEXITCODE -ne 0) { throw "VEA python cannot import open_clip." }

function Invoke-CheckedPython {
    param([string[]]$PythonArguments)
    & $Python @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($PythonArguments -join ' ')"
    }
}

function Assert-RunArtifacts {
    param([string]$RunName)
    $CheckCode = "import csv,math,torch; from pathlib import Path; run=r'$RunName'; checkpoint_path=Path('checkpoints')/run/'best.pt'; checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False); assert checkpoint.get('checkpoint_format')=='head_only_frozen_clip'; assert not any(key.startswith('clip_model.') for key in checkpoint['model_state_dict']); assert checkpoint_path.stat().st_size < 30*1024*1024; metrics=list(csv.DictReader(open(Path('results')/run/'metrics.csv',newline='',encoding='utf-8-sig'))); assert metrics and all(math.isfinite(float(value)) for row in metrics for value in row.values() if value!=''); predictions=list(csv.DictReader(open(Path('results')/run/'predictions.csv',newline='',encoding='utf-8-sig'))); required={'adapter_residual_abs_mean','adapter_residual_norm','layer_scale_mean'}; assert predictions and required.issubset(predictions[0]); assert all(math.isfinite(float(value)) for row in predictions for key,value in row.items() if key!='path' and value!='')"
    & $Python -c $CheckCode
    if ($LASTEXITCODE -ne 0) {
        throw "Artifact validation failed for $RunName."
    }
}

Invoke-CheckedPython @("-m", "compileall", "src", "train.py", "evaluate.py", "scripts", "tests")
Invoke-CheckedPython @("scripts/inspect_openclip_visual.py")
Invoke-CheckedPython @("-m", "unittest", "tests.test_internal_spectral_adapters", "-v")

$Cases = @(
    @{ Config = "configs/roi_spatial_token_adapter.yaml"; Smoke = "smoke_spatial_token_adapter"; Run = "roi_spatial_token_adapter_seed42" },
    @{ Config = "configs/roi_spectral_global_filter_adapter.yaml"; Smoke = "smoke_spectral_global_filter_adapter"; Run = "roi_spectral_global_filter_adapter_seed42" },
    @{ Config = "configs/roi_spectral_factorized_filter_adapter.yaml"; Smoke = "smoke_spectral_factorized_filter_adapter"; Run = "roi_spectral_factorized_filter_adapter_seed42" },
    @{ Config = "configs/roi_wavelet_token_adapter.yaml"; Smoke = "smoke_wavelet_token_adapter"; Run = "roi_wavelet_token_adapter_seed42" }
)

foreach ($Case in $Cases) {
    Invoke-CheckedPython @(
        "train.py", "--config", $Case.Config, "--epochs", "2",
        "--max_train_samples", "64", "--max_test_samples", "64",
        "--num_workers", "0", "--run_name", $Case.Smoke
    )
    Invoke-CheckedPython @(
        "evaluate.py", "--config", $Case.Config,
        "--ckpt", "checkpoints/$($Case.Smoke)/best.pt",
        "--run_name", $Case.Smoke, "--num_workers", "0"
    )
    Assert-RunArtifacts $Case.Smoke
}

foreach ($Case in $Cases) {
    Invoke-CheckedPython @(
        "train.py", "--config", $Case.Config,
        "--seed", "42", "--run_name", $Case.Run
    )
    Invoke-CheckedPython @(
        "evaluate.py", "--config", $Case.Config,
        "--ckpt", "checkpoints/$($Case.Run)/best.pt",
        "--run_name", $Case.Run
    )
    Assert-RunArtifacts $Case.Run
}

Invoke-CheckedPython @("scripts/summarize_phase7_internal_adapters.py")
