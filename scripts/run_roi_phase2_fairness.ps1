$ErrorActionPreference = "Stop"

conda activate VEA
if ($LASTEXITCODE -ne 0) {
    throw "Failed to activate conda environment VEA."
}

foreach ($seed in 42, 43, 44) {
    $baselineRun = "roi_clip_baseline_lr5e4_seed$seed"
    python train.py --config configs/roi_clip_baseline.yaml --head_lr 5.0e-4 --seed $seed --run_name $baselineRun
    if ($LASTEXITCODE -ne 0) { throw "$baselineRun training failed." }
    python evaluate.py --config configs/roi_clip_baseline.yaml --ckpt "checkpoints/$baselineRun/best.pt" --run_name $baselineRun
    if ($LASTEXITCODE -ne 0) { throw "$baselineRun evaluation failed." }

    $concatRun = "roi_clip_fft_concat_seed$seed"
    python train.py --config configs/roi_clip_fft_concat.yaml --seed $seed --run_name $concatRun
    if ($LASTEXITCODE -ne 0) { throw "$concatRun training failed." }
    python evaluate.py --config configs/roi_clip_fft_concat.yaml --ckpt "checkpoints/$concatRun/best.pt" --run_name $concatRun
    if ($LASTEXITCODE -ne 0) { throw "$concatRun evaluation failed." }

    $filmRun = "roi_affectspectrum_film_seed$seed"
    python train.py --config configs/roi_affectspectrum_film.yaml --seed $seed --run_name $filmRun
    if ($LASTEXITCODE -ne 0) { throw "$filmRun training failed." }
    python evaluate.py --config configs/roi_affectspectrum_film.yaml --ckpt "checkpoints/$filmRun/best.pt" --run_name $filmRun
    if ($LASTEXITCODE -ne 0) { throw "$filmRun evaluation failed." }
}

python scripts/summarize_phase2_fairness.py
if ($LASTEXITCODE -ne 0) { throw "Fairness summary failed." }
