# Phase 7 Internal Spectral Adapter Sprint

Manual OpenCLIP parity: passed (max_abs_diff=0.0000000000e+00, mean_abs_diff=0.0000000000e+00).
Spatial control best_acc=0.654882, macro_f1=0.654244.

- spatial_token_adapter: best_acc=0.654882, macro_f1=0.654244, delta_vs_spatial=+0.000000, mean_residual=6.447291e-01, mean_layer_scale=1.808571e-02.
- spectral_global_filter_adapter: best_acc=0.668350, macro_f1=0.670097, delta_vs_spatial=+0.013468, mean_residual=4.840184e-01, mean_layer_scale=3.323005e-02.
- spectral_factorized_filter_adapter: best_acc=0.688552, macro_f1=0.692077, delta_vs_spatial=+0.033670, mean_residual=3.089997e-01, mean_layer_scale=3.673920e-02.
- wavelet_token_adapter: best_acc=0.676768, macro_f1=0.680727, delta_vs_spatial=+0.021886, mean_residual=4.628502e-01, mean_layer_scale=2.918419e-02.

Best frequency model: spectral_factorized_filter_adapter at 0.688552.
It does exceed the spatial control.
It does not reach concat/gated seed42 (0.705387).
Frequency response magnitudes: spectral_global_filter_adapter=7.058494e-02, spectral_factorized_filter_adapter=4.328405e-02, wavelet_token_adapter=5.531639e-02.
Adapter residuals are non-zero by the 1e-4 mean threshold.
Layer scales are not uniformly near initialization.
Frequency-specific gains: spectral_global_filter_adapter=+0.013468, spectral_factorized_filter_adapter=+0.033670, wavelet_token_adapter=+0.021886.

Decision: ABANDON_FREQUENCY_MAINLINE
