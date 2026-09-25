# Amazon ML Challenge 2026 — Business Entity Resolution

This project connects your existing `clean_er_data.py` outputs to model-ready pair tables. **It contains no LightGBM training or model inference code.** After you train a model separately, the evaluation and submission modules accept its probabilities.

Pipeline:

**Cleaned data → reference index → blocking → final candidate pairs → ground-truth labels → pairwise features → train/validation tables → [your model] → [your probabilities] → macro F₀.₅ / submission.**

## 0. Clean the original dataset

The repository includes all code built so far: optional sampling, cleaning, blocking,
labeling, features, table export, and probability evaluation/submission formatting.
LightGBM model training and inference are not yet implemented.

Place `ML_Dataset.zip` locally beside `run.py`, then run:

```bash
python clean_er_data.py --input "ML_Dataset.zip" --output "cleaned_data"
```

To optionally create a small debugging sample first:

```bash
python make_er_sample.py --input "ML_Dataset.zip" --output "er_sample.zip"
python clean_er_data.py --input "er_sample.zip" --output "cleaned_sample"
```

Original datasets, labels, generated feature tables, credentials and model artifacts
are excluded from Git. No challenge data is included in this repository.

## 1. Setup

Use Python **3.11–3.13**, preferably a fresh virtual environment. No GPU is required. SQLite must include FTS5; most standard Python builds do. An unavailable FTS5 extension produces `no such module: fts5` during indexing.

Clone this repository and open a terminal **inside the repository folder**, where `run.py` is located. The commands assume cleaned data is generated inside that folder:

```bash
python -m pip install -r requirements.txt
```

`--cleaned` expects the CLEANER OUTPUT FOLDER, not `ML_Dataset.zip`. Run the included cleaning script first. Files can be `.tsv.gz` or `.tsv`; exactly one version of each must exist.

Required inputs:

- `train/train_source1.tsv.gz`, `train/train_source2.tsv.gz`, `train/train_source3.tsv.gz`
- `train/train_ground_truth.tsv`
- For test preparation: `test/test_source1.tsv.gz`, `test/test_source2.tsv.gz`, `test/test_source3.tsv.gz`

The cleaner's additional columns are required. They are checked before indexing.

## 2. Run the connected pipeline

First run on your already-cleaned **small sample**, in a separate work folder:

```bash
python run.py prepare --cleaned "cleaned_sample" --work "work_sample" --split train
python run.py prepare --cleaned "cleaned_sample" --work "work_sample" --split test
```

Then process the original cleaned challenge dataset:

```bash
python run.py prepare --cleaned "cleaned_data" --work "work_full" --split train
python run.py prepare --cleaned "cleaned_data" --work "work_full" --split test
```

In a notebook, prefix a command with `!`, for example:

```python
!python run.py prepare --cleaned "cleaned_data" --work "work_full" --split train
```

These commands stop at feature tables. They do not train a model.

For a pilot against the **full reference pool**:

```bash
python run.py prepare --cleaned "cleaned_data" --work "work_pilot" --split train --max-queries 1000
```

`--max-queries` limits Source 1 only. Indexing still reads every Source 2/3 record so retrieval difficulty reflects the full pool. This is a smoke test of the first N S1 records, not a representative validation sample. Use a NEW work folder when removing the limit or changing the configuration.

On rerun with the SAME command/config/input/code, completed stages are skipped. An interrupted stage restarts from its beginning; indexing is not resumed halfway through. Changed inputs/configuration/code require a new `--work` folder. Do not manually mix outputs from different runs.

## 3. Modules in order

| Order | Python module under `src/er_pipeline/` | Responsibility | Main artifact |
|---|---|---|---|
| 1 | `indexing.py` | Stream cleaned rows into SQLite; index S2/S3; fit training reference TF-IDF | `index.sqlite`, `tfidf.npz` |
| 2 | `blocking.py` | A–F blocks, union, deduplication, ranking and final cap | `candidate_pairs.parquet`, `queries.tsv.gz` |
| 3 | `labeling.py` | Join actual candidates to ground truth; assign S1-level split; measure recall | `labeled_pairs.parquet`, `candidate_recall.json` |
| 4 | `features.py` | Compute your pairwise features with explicit missingness | `pair_features.parquet` |
| 5 | `tables.py` | Export train/validation/test tables and exact final candidate lists | `train_features.parquet`, `validation_features.parquet`, `test_features.parquet` |
| 6, later | **Your model — intentionally not included** | Fit classifier and calculate probabilities | Your probability file |
| 7, later | `evaluation.py` | Tune threshold using per-S1 macro F₀.₅; format supplied test probabilities | `validation_f05.json`, `matching_results.tsv` |

Supporting modules: `common.py` (streaming I/O/configuration), `text_features.py` (TF-IDF and text keys), `cli.py` (orchestration).

You can run stages individually with the SAME options/work folder:

```bash
python run.py prepare --cleaned "cleaned_data" --work "work_full" --split train --stage index
python run.py prepare --cleaned "cleaned_data" --work "work_full" --split train --stage block
python run.py prepare --cleaned "cleaned_data" --work "work_full" --split train --stage label
python run.py prepare --cleaned "cleaned_data" --work "work_full" --split train --stage features
python run.py prepare --cleaned "cleaned_data" --work "work_full" --split train --stage export
```

Test runs skip labeling. They reuse `work_full/train/tfidf.npz`; they never refit TF-IDF on test text. Test retrieval still indexes test S2/S3 because those are the records being searched.

## 4. The blocking mechanism

Every block searches S2 AND S3. Known countries restrict retrieval to the same country plus references with unknown country. A query with unknown country searches all countries; it is not dropped. France is included.

| Channel | Keys / retrieval |
|---|---|
| A | Exact basic name, suffix-free Latin-folded name, or sorted name tokens |
| B | First three characters of compact suffix-free name |
| C | State + postal code; city + address token; city + first three name tokens |
| D | Zero-normalized house number + name prefix |
| E | Name words and within-word character trigrams → indexed shortlist → hashed TF-IDF cosine reranking |
| F | Address words and character trigrams → indexed shortlist → address TF-IDF reranking |

Country applies to every channel. Missing key components do not become shared empty-string blocks. B is only a prefix heuristic; it is NOT a substitute for E. Whole-word reordering is handled by sorted-name A, bag-of-word/within-word grams E, and pair features.

Candidate generation:

1. Retrieve bounded candidates from each channel.
2. Union and deduplicate by record ID. Identical text with different IDs remains separate.
3. Rank by `0.55 × name cosine + 0.30 × address cosine + 0.10 × exact basic name + 0.01 × channel count`.
4. Apply the configurable final cap. The result is the ACTUAL set passed to feature extraction and, later, your model.

Missing similarities contribute zero **only to this retrieval ranking heuristic**. Their model feature values remain NaN, with missingness flags.

Broad-block protection: if an exact/prefix/component block exceeds `block_limit`, it is searched using lexical relevance within that block. It is never accepted as an unbounded Cartesian product. Overflow and final-cap counts are reported. Exact-name blocks can also be capped; inspect recall before choosing limits.

E/F are **approximate retrieval**, not exhaustive global TF-IDF nearest neighbours. SQLite FTS5 BM25 finds a bounded shortlist and TF-IDF reranks it. Some true neighbours may be outside that shortlist. Compare candidate recall while adjusting `lexical_pool`, `query_terms`, `lexical_top_k` and `max_candidates`.

## 5. Ground-truth labels and validation

For each actual candidate `(Source 1 ID, candidate ID)`:

- `label = 1` if the ID occurs in that S1 ground-truth list.
- `label = 0` otherwise.

All generated negatives are retained; there is no hidden negative downsampling. Known positive pairs missed by blocking are **not inserted** into candidate/validation tables. Missing ground-truth rows or referenced records are input errors, not assumed negatives.

Each S1 ID is deterministically assigned to approximately 80% train / 20% validation using a seeded hash including country and singleton status. This is approximate stratification, not an exact quota per stratum. **All pairs belonging to an S1 stay together.** Check group counts, especially on tiny samples. Train and validation retrieve against the same full training S2/S3 pool, matching the reference-search task. This is an S1 holdout, not a guarantee that every business name/token is unseen.

TF-IDF statistics use only training reference (S2/S3) text. They never use labels, held-out S1 text or test text. No business identity lookup, geocoding or external data is used.

`queries.tsv.gz` includes every selected S1, even with zero candidates. `query_labels.tsv.gz` additionally contains the split, full truth and hit counts. Keep these files: pair tables cannot represent zero-candidate queries on their own.

`candidate_recall.json` reports overall, train/validation and country-specific:

- Candidate recall = retrieved true links / ALL true links.
- Fraction of non-singletons with all true matches retrieved.
- Positive and negative pair counts.
- Oracle macro F₀.₅ ceiling: best possible score if every retrieved true match is selected and every false candidate rejected. **This is not a trained model score.**

## 6. Pairwise features

Use the exact `features` list in `feature_columns.json` when you train later. Exclude IDs, `dataset_split` and `label` from model inputs. `label` is your target.

| Features | Definition |
|---|---|
| `name_levenshtein_ratio`, `address_levenshtein` | Normalized Levenshtein similarity: 1 − edit distance / maximum length |
| `name_jaro_winkler` | RapidFuzz normalized Jaro–Winkler similarity |
| `name_token_jaccard`, `address_token_jaccard` | Intersection / union of token sets |
| `name_tfidf_cosine`, `address_tfidf_cosine` | Cosine of frozen hashed TF-IDF vectors |
| `name_char_ngram_similarity`, `address_char_similarity` | Jaccard of word-boundary character trigram sets |
| `name_exact`, `address_exact` | Equality of nonempty normalized strings |
| `name_without_suffix_similarity`, `name_expanded_similarity` | Normalized Levenshtein on those separate name views |
| `postal_code_exact`, `city_exact`, `state_exact`, `house_number_exact`, `country_exact` | Equality when both components are present |
| `name_address_similarity_product` | Name Levenshtein similarity × address Levenshtein similarity |
| `name_missing`, `address_missing` | 1 if either side is missing |
| `*_missing_left`, `*_missing_right`, `*_missing_both` | Separate missingness indicators |
| Component missing flags | 1 if either component is absent |
| `address_edit_distance` | Raw edit distance, unlike the normalized similarity column |
| Name/address length ratios | Shorter length / longer length |
| Retrieval score/rank/source and block flags | Additional label-free retrieval evidence |

Similarities/equalities involving missing text/components are **NaN**, even when both are missing. Two empty addresses are not an exact-address match. Do not replace these NaNs with zero.

TF-IDF uses word tokens plus within-word 3–5-grams, log term frequency `1 + log(tf)`, smoothed IDF `log((N+1)/(df+1))+1`, and L2 normalization. Names use the suffix-free Latin-folded view with a basic-name fallback; addresses use the normalized Latin-folded view. Hashing keeps IDF memory fixed; collisions are possible, so these cosines approximate an explicit-vocabulary TF-IDF representation. Retrieval token encoding itself is lossless UTF-8 hex, not hashed.

Unicode is retained. Character grams do not translate Hindi/Tamil/etc. into Latin. Cross-script matches may require address evidence or a later multilingual retrieval channel. The cleaner's city/state/postal/house columns remain heuristic evidence. House-number disagreement never automatically rejects a match.

## 7. Final outputs

Inside `work_full/train/`:

- **`train_features.parquet`** — model training table with target `label`.
- **`validation_features.parquet`** — held-out candidate-pair table with `label`.
- `feature_columns.json` — exact model-input list.
- `candidate_recall.json`, `blocking_report.json`, `tables_report.json` — quality and size checks.
- `query_labels.tsv.gz` — full truth including singleton/zero-candidate queries.

Inside `work_full/test/`:

- **`test_features.parquet`** — same numerical inputs, no target.
- **`candidate_pairs.tsv`** — challenge format: `source1_entity_id`, `candidate_entity_ids`; one row per S1, comma-separated IDs.

The `.parquet` candidate file is one row per pair; the `.tsv` candidate file is one row per S1. Do not confuse these two representations.

You do not need pandas for this pipeline. To inspect a few rows without loading a full table:

```python
import pyarrow.parquet as pq
file = pq.ParquetFile('work_full/train/train_features.parquet')
batch = next(file.iter_batches(batch_size=5))
print(batch.to_pylist())
```

## 8. Later: probabilities, F₀.₅ and submission

Your separately trained model must score **every row** of `validation_features.parquet` or `test_features.parquet`, preserving these keys. Save predictions as TSV (optionally gzipped) or Parquet:

| source1_entity_id | candidate_entity_id | match_probability |
|---|---|---|
| S1-... | S2-... | 0.93 |
| S1-... | S3-... | 0.14 |

These are illustrative values only. No probability file is produced by preparation.

Evaluate validation probabilities and compare thresholds:

```bash
python run.py evaluate --work "work_full" --probabilities "validation_probabilities.tsv"
```

Or provide a custom grid:

```bash
python run.py evaluate --work "work_full" --probabilities "validation_probabilities.tsv" --thresholds "0.7,0.8,0.85,0.9,0.95,0.98"
```

For each S1:

`F0.5 = 5*TP / (5*TP + 4*FP + FN)`.

Both truth and prediction empty → 1. Truth empty but prediction nonempty → 0. Average across **all validation S1**, including zero-candidate queries and true positives missed by blocking. This is not a global binary-pair F-score. The threshold selected on validation is a tuning result, not an unbiased final-test estimate.

After scoring test pairs, use your selected threshold (0.9 below is only an example):

```bash
python run.py submit --work "work_full" --probabilities "test_probabilities.tsv" --threshold 0.9
```

This creates `work_full/test/matching_results.tsv`. It permits zero, one or many matches, rejects missing/duplicate/unexpected probability rows, ensures matches come from actual candidates, and refuses incomplete test-query runs.

Run the official `utils/validate_submission.py` supplied with the challenge against `matching_results.tsv`, `candidate_pairs.tsv`, and the original test source files before uploading. This project supplies data preparation and formatting, not a trained submission model.

## 9. Resource controls and tests

`config.json` contains the defaults. To customize, edit a copy and pass `--config your_config.json` consistently for every stage/split in that work folder.

| Setting | Default | Meaning |
|---|---:|---|
| `max_candidates` | 80 | Final maximum candidates per S1 |
| `block_limit` | 100 | Maximum results per A–D channel |
| `lexical_pool` | 120 | Indexed shortlist size for E/F before TF-IDF reranking |
| `lexical_top_k` | 25 | Retained candidates per E/F channel |
| `query_terms` | 16 | Rare lexical tokens used in each search |
| `hash_bits` | 18 | TF-IDF dimension = 2^18 per field |
| `row_group_size` | 10000 | Bounded buffered rows for Parquet output |
| `sqlite_cache_mb` | 128 | Cache budget per SQLite connection |

There is no all-pairs matrix and no full in-memory reference table. The index and intermediate/output tables still require substantial disk space and CPU time. Bounded RAM does not make millions of queries cheap: at 80 candidates, 2.2 million S1 records can create roughly **176 million pair rows**. Lower caps reduce cost but may reduce recall. Defaults are starting values, not competition-tuned optimums. Multiple intermediate tables are intentionally retained for auditing; plan disk space for them and the index. The full dataset has not been benchmarked on your machine.

Run the included synthetic correctness tests:

```bash
python -m unittest discover -s tests -v
```

Tests cover connected stages, country separation, duplicate-looking IDs, oversized blocks, missing values, word reordering, missed positives, S1 split separation, singleton/zero-candidate metric behavior, and rejection of missing probability rows. `SAMPLE_VERIFICATION.md` records the actual uploaded-sample checks.

API references used for implementation: [SQLite FTS5](https://www.sqlite.org/fts5.html), [RapidFuzz Levenshtein](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html), [RapidFuzz Jaro–Winkler](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/JaroWinkler.html), [PyArrow ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
