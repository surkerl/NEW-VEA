# Phase 7 Internal Spectral Adapter Sprint

Manual OpenCLIP parity: passed (max_abs_diff=0.0000000000e+00, mean_abs_diff=0.0000000000e+00).
Spatial control best_acc=0.648148, macro_f1=0.649662.

- spatial_token_adapter: best_acc=0.648148, macro_f1=0.649662, delta_vs_spatial=+0.000000, mean_residual=6.728876e-01, mean_layer_scale=1.393480e-02.
- spectral_global_filter_adapter: best_acc=0.668350, macro_f1=0.672112, delta_vs_spatial=+0.020202, mean_residual=4.891311e-01, mean_layer_scale=3.166002e-02.
- spectral_factorized_filter_adapter: best_acc=0.698653, macro_f1=0.701415, delta_vs_spatial=+0.050505, mean_residual=2.965496e-01, mean_layer_scale=3.062632e-02.
- wavelet_token_adapter: best_acc=0.680135, macro_f1=0.684019, delta_vs_spatial=+0.031987, mean_residual=4.416959e-01, mean_layer_scale=2.486743e-02.

Best frequency model: spectral_factorized_filter_adapter at 0.698653.
It does exceed the spatial control.
It does not reach concat/gated seed42 (0.705387).
Frequency response magnitudes: spectral_global_filter_adapter=6.922686e-02, spectral_factorized_filter_adapter=5.743279e-02, wavelet_token_adapter=5.655397e-02.
Adapter residuals are non-zero by the 1e-4 mean threshold.
Layer scales are not uniformly near initialization.
Frequency-specific gains: spectral_global_filter_adapter=+0.020202, spectral_factorized_filter_adapter=+0.050505, wavelet_token_adapter=+0.031987.

Decision: ABANDON_FREQUENCY_MAINLINE
