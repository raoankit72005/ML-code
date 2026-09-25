# Verification

The synthetic tests cover:

- A real CPU LightGBM fit using the new disk-backed Sequence, with all training and validation pairs retained despite deliberately tiny legacy pair caps.
- Float32 cache reload and equality with the original Parquet features/labels, including NaN.
- Explicit refusal when estimated RAM or GPU memory exceeds available budgets.
- End-to-end `full` execution: original-format cleaning, retrieval, labels, features, all-pair fitting, threshold selection, test prediction, submission and original-ID audit.
- Repeating the full command and reusing completed stages.
- Missing original test S1 rows, mismatched probability order, and sampled-input rejection.
- Zero-candidate singletons, blocked-out positives, F0.5 semantics, Unicode, and S1-level validation splitting.

Run `python -m unittest discover -s tests -v`.

A CUDA integration test is gated by `RUN_CUDA_TESTS=1`. It must run on the target NVIDIA GPU with the CUDA-enabled LightGBM build. Development verification used CPU execution because no NVIDIA GPU was available. CUDA throughput, native peak VRAM, the full original dataset, and leaderboard improvements have not been measured here.

The previous 500-query sampled score was a historical smoke test. It is deliberately not advertised as the performance of this full-data version. Larger retrieval budgets and new features require fresh preparation and training.
