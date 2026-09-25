# Amazon ML Challenge 2026 — entity resolution

Match each S1 business to zero or more S2/S3 records. The full-data workflow uses the **original dataset**, all reference records, every non-holdout training pair, and CUDA LightGBM. It generates both required TSV files and checks them against the original test S1 IDs.

## Start on SageMaker

Use a **Linux NVIDIA GPU app with a CUDA development toolkit (`nvcc`), `g++`, and Python 3.11+**. A GPU instance alone does not install the CUDA-enabled LightGBM package. See [the complete SageMaker guide](docs/SAGEMAKER_GPU.md) for installation, sizing, restarting, and troubleshooting.

From your activated Python environment, in the repository:

```bash
git pull --ff-only
bash scripts/setup_gpu.sh
python run.py full --input ML_Dataset.zip --work work_full_gpu
```

Use the original `ML_Dataset.zip`. Do not run `make_er_sample.py`. Use a **new work directory** for this version; old prepared features/models have different schemas and retrieval behavior.

To train and validate first, then generate the complete submission later:

```bash
python run.py full --input ML_Dataset.zip --work work_full_gpu --skip-test
python run.py full --input ML_Dataset.zip --work work_full_gpu
```

The second command skips completed cleaning/preparation/training stages with matching fingerprints. It still needs the original dataset. Interrupted preparation stages restart that stage; interrupted training restarts fitting and reuses completed disk caches. It does not resume from a partially fitted tree ensemble.

## What “full data” means

- Every original training S1 is processed, with all original S2/S3 records available for retrieval.
- S1 queries are divided deterministically into approximately 80% training and 20% validation. All candidate pairs from each partition are retained. No pair cap or negative sampling applies in `full` mode.
- Validation labels remain held out of gradient training. This workflow does not refit on the holdout after tuning.
- Blocking deliberately limits candidate pairs. Full data does **not** mean the impossible Cartesian product of all business records. Missed true links remain visible in candidate recall and F0.5.
- All original test S1 queries, including zero-match cases and France, receive output rows. Test reference pools are never sampled.

## Memory and runtime

`DiskSequence` streams Parquet into reusable float32 files and gives LightGBM bounded batches instead of allocating one dense matrix containing every pair. Batch sizes adapt to available RAM, including Linux cgroup limits. Before fitting, the pipeline writes a conservative RAM/VRAM sizing report and refuses configurations that exceed its budget. It never silently switches to CPU or samples rows.

**LightGBM still holds binned datasets and training buffers in RAM/VRAM.** It is not an external-memory boosting engine. The estimate is a heuristic, not an OOM guarantee. If the report rejects the run, use an instance with more RAM/VRAM; reducing Python batch size alone cannot fix the resident dataset size. Current CUDA training uses one GPU, not pooled VRAM across multiple GPUs.

Cleaning, SQLite retrieval, feature extraction, and prediction remain CPU/disk work. GPU acceleration applies to fitting LightGBM; it will not make the entire pipeline GPU accelerated. Full-scale throughput has not been benchmarked on your dataset.

Disk headroom checks stop writes before the filesystem is nearly full. Full original source indexing and candidate features can take far more space than the compressed ZIP. Keep substantial free disk and inspect `df -h` throughout preparation. No raw data or model outputs are committed to GitHub.

## Retrieval and F0.5 changes

The union of exact names, name prefixes, location/house blocks, TF-IDF reranked name searches, and address searches now includes:

- Separate whole-word searches in addition to character n-grams.
- Postal-code blocking without requiring a successfully extracted state.
- Exact expanded and suffix-free name blocks.
- A small reserved quota for strong address/name candidates, preventing the combined rank from discarding all address-led matches.
- Twice the lexical retrieval budget and candidate cap for India in the full-data preparation config (up to 160 candidates; other countries up to 80).
- Token-sort name/address features for reordered text, preserving NaN for missing fields.
- A finer validation threshold grid, with F0.5 computed over all validation S1 queries and all true labels, including links missed by blocking.

These changes target the weaker India retrieval recall from the earlier run. They are **not a demonstrated score improvement**. Check `train/candidate_recall.json`, country-level F0.5, candidate count, and timing. Larger candidate sets increase memory and runtime; tune only using validation and rebuild in a new work folder.

## Useful commands

```bash
# Check the actual CUDA installation before expensive preparation.
python run.py doctor

# After training tables exist, estimate resident RAM/VRAM without fitting.
python run.py estimate --work work_full_gpu

# Recheck submission coverage against the ORIGINAL dataset.
python run.py validate-submission --work work_full_gpu --original ML_Dataset.zip

# CPU/unit integration tests; CUDA test is skipped unless explicitly enabled.
python -m unittest discover -s tests -v
RUN_CUDA_TESTS=1 python -m unittest discover -s tests -v
```

The original modular commands remain available through `python run.py --help`. `baseline` and `lgbm_config.json` retain the earlier capped CPU workflow for debugging. Use **`full`** and **`configs/full_gpu.json`** for your requested training run.

## Outputs

| File | Purpose |
| --- | --- |
| `work_full_gpu/train/candidate_recall.json` | Retrieval recall and oracle F0.5 ceiling |
| `work_full_gpu/model/resource_plan.json` | Estimated training RAM/VRAM and chosen loading batch size |
| `work_full_gpu/model/lightgbm_model.txt` | Best early-stopping iteration |
| `work_full_gpu/model/model_metadata.json` | Counts, backend config, threshold, hashes, and validation result |
| `work_full_gpu/model/validation_f05.json` | Full validation threshold/country scores |
| `work_full_gpu/test/matching_results.tsv` | Final predicted matches |
| `work_full_gpu/test/candidate_pairs.tsv` | Actual final candidate lists |
| `work_full_gpu/test/submission_coverage.json` | Exact original test S1 coverage check |

Run the challenge's supplied validator too, including reference-ID checks, before uploading the two TSV files. The original-input audit assumes the supplied ZIP really is the original challenge release; an arbitrary unmarked subset cannot be identified without the official dataset.

## Modules

`clean_er_data.py` → `indexing.py` / `blocking.py` → `labeling.py` → `features.py` / `tables.py` → `disk_training.py` / `modeling.py` → `evaluation.py` / `coverage.py`.

- `configs/`: full-data preparation and GPU training settings.
- `scripts/`: GPU installation helper.
- `docs/`: SageMaker instructions and verification notes.
- `tests/`: synthetic regression and full-command integration tests.
- `make_er_sample.py`: optional development utility; never used by `full`.

[Verification and limitations](docs/TESTING.md). License: MIT.
