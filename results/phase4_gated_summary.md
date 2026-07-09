# Phase 4 AffectSpectrum-Gated v1 Summary

Gated mean best_acc: 0.704826
Gated mean macro_f1: 0.703838
Mean delta vs baseline_lr5e4: +0.006734
Mean delta vs concat: -0.000561
Win count vs baseline_lr5e4: 2/3
Win count vs concat: 1/3

Per-seed clean results:
- seed42: best_acc=0.705387, best_epoch=2, macro_f1=0.701815, mean_test_gate=0.0004
- seed43: best_acc=0.696970, best_epoch=7, macro_f1=0.698846, mean_test_gate=0.3215
- seed44: best_acc=0.712121, best_epoch=4, macro_f1=0.710854, mean_test_gate=0.0031

Baseline vs gated mean net correction: +4.00
Concat vs gated mean net correction: -0.33

Mean gate: 0.1083. Gate is not collapsed to 0 or 1.
Response auxiliary loss trend: seed42: train_response_loss 1.7965->1.7441; seed43: train_response_loss 1.7987->1.7419; seed44: train_response_loss 1.7915->1.7441.

Gated stronger than concat under: amplitude_noise_0.05 (+0.0123), amplitude_noise_0.10 (+0.0034), band_0.30_0.50 (+0.0022), blur_heavy (+0.0426), blur_light (+0.1678), downsample_x2 (+0.0073), downsample_x4 (+0.1538), high_0.30 (+0.0006), low_0.15 (+0.1510), low_0.30 (+0.2104), low_0.50 (+0.0107), phase_noise_0.05 (+0.0022)

Gated weaker than concat under: band_0.50_0.75 (-0.0034), high_0.50 (-0.0045), phase_noise_0.10 (-0.0017)

Low/blur/downsample comparison vs concat: blur_heavy (+0.0426), blur_light (+0.1678), downsample_x2 (+0.0073), downsample_x4 (+0.1538), low_0.15 (+0.1510), low_0.30 (+0.2104), low_0.50 (+0.0107)

Recommendation: proceed to Phase 5 only if the clean and perturbation summaries show a useful tradeoff against concat; otherwise tune the gate regularization and response loss before adding log-spaced spectrum or local spectral tokens.
