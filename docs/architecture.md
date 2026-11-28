# Architecture

## Contracts

Contracts are the file formats and interfaces that cross between Arvind's and Pratyush's halves of the code. Each one is agreed in its own PR. Once a contract is merged, it changes only through another PR that both people approve.

### C1. Split manifest

**Produced by:** `argus splits` (A1.2, Arvind).
**Read by:** feature extraction, memory bank building, scoring (Arvind); calibration, evaluation, stress tests (Pratyush).

A split manifest lists which image plays which role for one dataset, one category (MVTec AD) or modality (ISP-AD), and one seed. It is the only place roles are decided. No other code may pick images for a role on its own.

#### Location

```
artifacts/splits/{dataset}/{category}/seed{n}.csv
```

- `{dataset}`: `mvtec_ad` or `isp_ad`.
- `{category}`: the category or modality folder name, exactly as it appears under `data/{dataset}/` after A1.1 (for example `hazelnut`, `screw`).
- `{n}`: the actual seed value from the dataset config (for example `seed20260801.csv`), not a run index.

#### File format

- UTF-8 CSV, comma-separated, one header row, LF line endings, no index column.
- Rows are sorted by `role` (in the order `support`, `calib`, `eval_normal`, `test`) and then by `image_id`, so the same seed always gives a byte-identical file.

#### Columns

| Column | Type | Meaning |
|---|---|---|
| `image_id` | string | Image path relative to `ARGUS_DATA_DIR`, without the file extension, using `/` as separator. Example: `mvtec_ad/hazelnut/train/good/000`. The same image has the same id in every seed, so features are cached once per image. |
| `path` | string | Image path relative to `ARGUS_DATA_DIR`, with extension. |
| `role` | string | One of `support`, `calib`, `eval_normal`, `test` (see below). |
| `label` | int | `0` = good, `1` = defect. |
| `defect_type` | string | `good` for normal images, otherwise the dataset's defect folder name (for example `crack`). |
| `mask_path` | string | Ground-truth mask path relative to `ARGUS_DATA_DIR`. Empty for good images, and for defect images that have no mask. |

All paths are relative, so the same manifest works on both laptops and on Colab.

#### Roles

| Role | What it is for | Images |
|---|---|---|
| `support` | Builds the memory bank. Never scored. | Good images only. |
| `calib` | Sets the cut-off τ with `fit_threshold` (C4). | Good images only. |
| `eval_normal` | Extra good images for measuring FPR at τ. | Good images only. |
| `test` | The dataset's own test set. | Good and defect images. |

Where each role comes from:

- **MVTec AD:** `support`, `calib` and `eval_normal` are drawn from `train/good` using the seed and the counts in `configs/datasets/mvtec_ad.yaml` (0.4). `test` is the whole original `test/` folder (good and defect) and is the same in every seed.
- **ISP-AD:** `support` (300), `calib` (150) and `eval_normal` (200) are drawn from the modality's good images using the seed, following Deng's split. `test` holds all real defect images of that modality, and is the same in every seed. If the downloaded release turns out to ship its own good test images, they go in `test` as well; A1.1 confirms this.

#### How each role is used

| Measure | Rows used |
|---|---|
| Memory bank | `support` |
| Threshold τ | `calib` |
| FPR at τ | every `label = 0` row in `eval_normal` and `test` |
| Recall at τ | every `label = 1` row in `test` |
| Image AUROC, pixel AUROC, PRO | `test` rows only (comparable with the PatchCore and ASO papers). If `test` has no good images (ISP-AD), `eval_normal` rows are added as the normals. |

Test-set good images stay in `test` instead of being moved to `eval_normal`. That keeps AUROC computed on exactly the papers' test set, and FPR still uses every held-out good image.

#### Rules

Every manifest must satisfy these; A1.2's tests (`tests/test_splits.py`) check them, and any reader may assert them.

1. Each `image_id` appears at most once per file, so roles are disjoint within a seed.
2. Defect images (`label = 1`) only ever have role `test`.
3. `label = 0` if and only if `defect_type = good`; `mask_path` is empty for every `label = 0` row.
4. `test` rows are identical across all seeds of the same dataset and category.
5. The same config and seed produce a byte-identical file; different seeds give a different `support` / `calib` / `eval_normal` assignment.
6. Role counts match the dataset config.
7. Every `path` and non-empty `mask_path` exists under `ARGUS_DATA_DIR`.
8. Images in the dataset that are not listed are not used by any stage.

Evaluation code must not read `eval_normal` or `test` scores for a run until that run's threshold file exists (the test-set lock from the execution plan).

#### Example

`artifacts/splits/mvtec_ad/hazelnut/seed20260801.csv`, one row per role. Which train images land in which role depends on the seed.

```csv
image_id,path,role,label,defect_type,mask_path
mvtec_ad/hazelnut/train/good/017,mvtec_ad/hazelnut/train/good/017.png,support,0,good,
mvtec_ad/hazelnut/train/good/203,mvtec_ad/hazelnut/train/good/203.png,calib,0,good,
mvtec_ad/hazelnut/train/good/088,mvtec_ad/hazelnut/train/good/088.png,eval_normal,0,good,
mvtec_ad/hazelnut/test/crack/004,mvtec_ad/hazelnut/test/crack/004.png,test,1,crack,mvtec_ad/hazelnut/ground_truth/crack/004_mask.png
mvtec_ad/hazelnut/test/good/012,mvtec_ad/hazelnut/test/good/012.png,test,0,good,
```

### C2. Score files

**Produced by:** the pipeline runner (A2.6, Arvind) for clean runs, and the stress tests (P3.4, Pratyush) for perturbed runs.
**Read by:** the schema checker (P2.3), the results table (P2.2), pixel metrics (P2.1), stress analysis (P4.1) and spatial lift (P4.5).

A run is one dataset, category, memory setting and seed. Each run writes one folder holding its image scores, its heatmaps and a `run.json`. Analysis code only reads these folders and never writes into them.

#### Location

```
artifacts/scores/{dataset}/{category}/{setting}/seed{n}/
├── scores.parquet
├── run.json
└── heatmaps/{image_id}.npy
```

- `{dataset}`, `{category}` and `seed{n}`: exactly as in C1, so a run folder and its split manifest share the same names.
- `{setting}`: `m100` (full memory), `m10` (10% coreset), `m1` (1% coreset) or `aso` (ASO's automatic size).
- `{image_id}`: the C1 id, so it contains `/` and the heatmaps sit in nested folders (for example `heatmaps/mvtec_ad/hazelnut/test/crack/004.npy`).

Every path inside a run folder is relative to that folder, so a run folder can be moved (for example to Google Drive) and still work.

#### `scores.parquet`

One row per scored image. Written with pyarrow, without a pandas index, and no column may be null.

| Column | Type | Meaning |
|---|---|---|
| `image_id` | string | C1 image id. Unique within the file. |
| `role` | string | `calib`, `eval_normal` or `test`. |
| `label` | int8 | `0` = good, `1` = defect. |
| `defect_type` | string | As in C1 (`good` for normal images). |
| `score` | float32 | Image anomaly score: the largest patch score. Higher means more anomalous; an image is flagged when `score > tau` (C4). Always finite. |
| `heatmap_path` | string | Heatmap file, relative to the run folder: `heatmaps/{image_id}.npy`. |
| `mask_path` | string | As in C1: relative to `ARGUS_DATA_DIR`, and an empty string (not null) for good images or defects without a mask. |

Rows:

- A clean run scores every `calib`, `eval_normal` and `test` row of its C1 manifest, and nothing else. `support` images are never scored.
- `role`, `label`, `defect_type` and `mask_path` are copied unchanged from the manifest.
- Rows are sorted by `role` (`calib`, `eval_normal`, `test`) and then by `image_id`.

#### Heatmaps

- One `.npy` file per row, float16, a 2-D array of shape `input_size` from `run.json` (for example 224 × 224).
- Values are patch scores, in the same units as `score`, after upsampling to input size and Gaussian smoothing (σ = 4) as in A2.4.
- Because of that smoothing, `heatmap.max()` is usually lower than `score`. Readers must not assume they are equal.
- The heatmap covers the model's input frame, not the original image. To compare it with a mask, load the mask at `mask_path` and apply the same resize and crop as the run's `preprocess` preset, using nearest-neighbour resampling, then treat any value above 0 as defect. A1.2's `argus.data.transforms` provides this mask transform for each preset, so both halves align masks the same way.

#### `run.json`

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | C2 version; `1` for this spec. Bumped by any change to this contract. |
| `config_hash` | string | Hash of the run's full config (A2.6). Same config, same hash. |
| `dataset`, `category`, `setting` | string | As in the folder path. |
| `seed` | int | As in the folder path. |
| `split_manifest` | string | C1 manifest used, relative to `ARGUS_ARTIFACTS_DIR`. |
| `preprocess` | string | Preprocessing preset (`patchcore`: resize 256, centre-crop 224; `deng`: resize to 256 × 256, no crop). |
| `input_size` | [int, int] | Model input height and width; also the heatmap shape. |
| `coreset_size` | int | Number of patches in the memory bank. |
| `coreset_fraction` | float | `coreset_size` divided by the number of support patches; `1.0` for `m100`. |
| `bank_mb` | float | Size of the memory bank as held for search, in MiB (bytes / 2²⁰). |
| `setup_s` | float | Seconds of greedy coreset selection needed to reach `coreset_size`, read from the per-step timings of the shared greedy run (A2.1). `0` for `m100`. Excludes support feature extraction, which is the same for every setting and is cached. |
| `ms_per_image` | float | Mean milliseconds per image from a preprocessed input tensor to score plus heatmap (features, nearest-neighbour search, heatmap). Batch size 1, GPU-synchronised, first 5 images excluded as warm-up. |
| `device` | string | Where scoring ran: `cuda:<GPU name>` (for example `cuda:Tesla T4`), `mps` or `cpu`. |
| `git_commit` | string | Full commit hash the run used, with `-dirty` appended if there were uncommitted changes. |

Sample, for hazelnut at 10% memory:

```json
{
  "schema_version": 1,
  "config_hash": "3f9a1c07b2e4",
  "dataset": "mvtec_ad",
  "category": "hazelnut",
  "setting": "m10",
  "seed": 20260801,
  "split_manifest": "splits/mvtec_ad/hazelnut/seed20260801.csv",
  "preprocess": "patchcore",
  "input_size": [224, 224],
  "coreset_size": 15680,
  "coreset_fraction": 0.1,
  "bank_mb": 61.25,
  "setup_s": 41.8,
  "ms_per_image": 47.3,
  "device": "cuda:Tesla T4",
  "git_commit": "f8b8de5c0e7d4a3b9f1e2d6c5b4a39281706f5e4"
}
```

These numbers are only illustrative. 15,680 patches is 10% of 200 support images × 784 patches each, and 15,680 × 1,024 dimensions × 4 bytes = 61.25 MiB.

Example rows of `scores.parquet` from the same run:

| image_id | role | label | defect_type | score | heatmap_path | mask_path |
|---|---|---|---|---|---|---|
| mvtec_ad/hazelnut/train/good/203 | calib | 0 | good | 1.874 | heatmaps/mvtec_ad/hazelnut/train/good/203.npy | |
| mvtec_ad/hazelnut/train/good/088 | eval_normal | 0 | good | 1.912 | heatmaps/mvtec_ad/hazelnut/train/good/088.npy | |
| mvtec_ad/hazelnut/test/crack/004 | test | 1 | crack | 3.406 | heatmaps/mvtec_ad/hazelnut/test/crack/004.npy | mvtec_ad/hazelnut/ground_truth/crack/004_mask.png |

#### Stress-test runs

Stress tests (P3.4) write the same folder layout and schema under:

```
artifacts/scores_stress/{perturbation}/{severity}/{dataset}/{category}/{setting}/seed{n}/
```

- `{perturbation}`: the perturbation name P2.4 defines (for example `brightness`, `contrast`, `blur`, `jpeg`).
- `{severity}`: the integer level, starting at `1` for the mildest.
- Rows: only `eval_normal` and `test`. There are no `calib` rows, because the threshold is always the clean run's τ and is never refitted on perturbed images.
- The matching clean run is the same path with `scores_stress/{perturbation}/{severity}/` replaced by `scores/`.
- `run.json` has every key above plus `perturbation` (string) and `severity` (int). `coreset_size`, `coreset_fraction`, `bank_mb` and `setup_s` are copied from the clean run, because the same memory bank is used.

#### Rules

The schema checker (P2.3) enforces these, and `report.py` refuses a folder that fails them.

1. `scores.parquet` has exactly the columns above, with those types and no nulls.
2. `image_id` is unique; `role` is one of the allowed values for the run type (clean: `calib`, `eval_normal`, `test`; stress: `eval_normal`, `test`).
3. `label = 1` only for `role = test`; `label = 0` if and only if `defect_type = good`; `mask_path` is empty for every `label = 0` row.
4. For a clean run, the rows match the non-support rows of `split_manifest` exactly, including `role`, `label`, `defect_type` and `mask_path`.
5. Every `score` is finite.
6. Every `heatmap_path` exists, is float16 and has shape `input_size`.
7. `run.json` has every key listed, with `schema_version = 1`, and its `dataset`, `category`, `setting` and `seed` match the folder path.
