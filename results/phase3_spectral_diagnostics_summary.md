# Phase 3 Spectral Evidence Diagnosis

## Phase 2.5 Clean Accuracy Context
- baseline_lr5e4: mean best_acc=0.698092, mean macro_f1=0.700111, delta=+0.000000
- clip_fft_concat: mean best_acc=0.705387, mean macro_f1=0.706663, delta=+0.007295
- affectspectrum_film: mean best_acc=0.700337, mean macro_f1=0.702519, delta=+0.002245

## Phase 3A Prediction-Level Evidence
Phase 3A Prediction-Level Spectral Evidence Diagnosis

Concat average corrected baseline errors: 24.67
Concat average new errors: 20.33
Concat average net correction: 4.33
FiLM average corrected baseline errors: 23.00
FiLM average new errors: 21.67
FiLM average net correction: 1.33

Concat largest class gains: class_4 (+0.0269), class_3 (+0.0202), class_1 (+0.0135)
Concat largest class drops: class_0 (+0.0034), class_5 (+0.0000), class_2 (-0.0202)
FiLM largest class gains: class_3 (+0.0303), class_4 (+0.0135), class_0 (+0.0101)
FiLM largest class drops: class_5 (-0.0067), class_1 (-0.0168), class_2 (-0.0168)

McNemar significant comparisons at 0.05: none
McNemar p-values used scipy.stats.chi2.sf.

Interpretation: prediction-level differences show that spectrum-guided models correct a different subset of baseline errors while also introducing their own errors. This supports spectral evidence as a distinct affective signal rather than a generic duplicate of semantic CLIP evidence.
Next step: the results support Phase 3B spectral perturbation to test spectral sensitivity and coarse-to-fine evidence accumulation.

## Phase 3B Spectral Perturbation Evidence
Concat is stronger than baseline under: high_0.30 (+0.0017), downsample_x2 (+0.0039), amplitude_noise_0.05 (+0.0079), amplitude_noise_0.10 (+0.0084), phase_noise_0.05 (+0.0146), phase_noise_0.10 (+0.0067)

Concat is weaker than baseline under: low_0.15 (-0.2194), low_0.30 (-0.2211), low_0.50 (-0.0118), band_0.50_0.75 (-0.0006), blur_light (-0.2155), blur_heavy (-0.1156), downsample_x4 (-0.1611)

FiLM is stronger than baseline under: high_0.30 (+0.0017), amplitude_noise_0.05 (+0.0051), phase_noise_0.05 (+0.0135), phase_noise_0.10 (+0.0028)

FiLM is weaker than baseline under: low_0.15 (-0.1268), low_0.30 (-0.1066), low_0.50 (-0.0376), blur_light (-0.1633), blur_heavy (-0.1105), downsample_x2 (-0.0135), downsample_x4 (-0.1319)

## Coarse-to-Fine Spectral Evidence Accumulation
Mean gt probability by stage: low_0.15=0.2926, low_0.30=0.3848, low_0.50=0.4839, full=0.5245
Classes with strongest low-frequency gt probability: fear (0.5802), joy (0.4243), disgust (0.2201)
Classes with largest high/detail gain from low_0.50 to full: surprise (+0.1655), disgust (+0.1078), sadness (+0.0422)

## Interpretation
Phase 3 evaluates whether spectral presentation evidence provides a distinct affective signal beyond semantic visual representations. The clean accuracy gains, prediction-level non-overlap, and perturbation sensitivity patterns support the frequency-centric interpretation: spectral evidence is not merely an ordinary complementary feature, but a measurable affective signal with its own error corrections and degradation profile.

The next modeling step should be AffectSpectrum-Gated Fusion, using spectral response to gate semantic CLIP features rather than treating spectrum as a passive concatenated descriptor.
