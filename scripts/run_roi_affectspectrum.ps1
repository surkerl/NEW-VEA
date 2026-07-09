$ErrorActionPreference = "Stop"

conda activate VEA
if ($LASTEXITCODE -ne 0) {
    throw "Failed to activate conda environment VEA."
}

python train.py --config configs/roi_frequency_only.yaml
if ($LASTEXITCODE -ne 0) { throw "frequency_only training failed." }
python evaluate.py --config configs/roi_frequency_only.yaml --ckpt checkpoints/roi_frequency_only_seed42/best.pt --run_name roi_frequency_only_seed42
if ($LASTEXITCODE -ne 0) { throw "frequency_only evaluation failed." }

python train.py --config configs/roi_clip_fft_concat.yaml
if ($LASTEXITCODE -ne 0) { throw "clip_fft_concat training failed." }
python evaluate.py --config configs/roi_clip_fft_concat.yaml --ckpt checkpoints/roi_clip_fft_concat_seed42/best.pt --run_name roi_clip_fft_concat_seed42
if ($LASTEXITCODE -ne 0) { throw "clip_fft_concat evaluation failed." }

python train.py --config configs/roi_affectspectrum_film.yaml
if ($LASTEXITCODE -ne 0) { throw "affectspectrum_film training failed." }
python evaluate.py --config configs/roi_affectspectrum_film.yaml --ckpt checkpoints/roi_affectspectrum_film_seed42/best.pt --run_name roi_affectspectrum_film_seed42
if ($LASTEXITCODE -ne 0) { throw "affectspectrum_film evaluation failed." }
