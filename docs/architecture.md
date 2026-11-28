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

A run is one dataset, category, memory setting and seed. Each run writes one folder holding its image scores, its heatmaps and a `run.json`. Analysis code never changes these files. The only thing it adds to a run folder is threshold files under `thresholds/` (C4).

#### Location

```
artifacts/scores/{dataset}/{category}/{setting}/seed{n}/
├── scores.parquet
├── run.json
├── heatmaps/{image_id}.npy
└── thresholds/          (added by calibration, C4)
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
- Rows: only `eval_normal` and `test`. There are no `calib` rows, because the threshold is always the clean run's τ and is never refitted on perturbed images. Stress folders therefore have no `thresholds/`; their τ is read from the clean run's `thresholds/`.
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

### C3. Detector interface

**Provided by:** `argus.load_detector` and the `Detector` class (A2.5, Arvind).
**Used by:** the backend (P1.6 with a stub, P3.2 with real banks) and the stress tests (P3.4, Pratyush).

C3 is the only thing code outside the detector needs to score images. Callers never build memory banks, never preprocess images and never read bank files directly.

#### Interface

```python
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


def load_detector(path: str | Path, device: str = "auto") -> "Detector":
    """Load a saved memory bank and its preprocessing config."""


class Detector(Protocol):
    @property
    def info(self) -> dict: ...

    def score(self, image: Image.Image) -> tuple[float, np.ndarray]: ...

    def score_batch(
        self, images: list[Image.Image], batch_size: int = 16
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def input_view(self, image: Image.Image) -> Image.Image: ...
```

`load_detector` and `Detector` are importable from `argus` itself. P1.6's `StubDetector` implements the same four members with the same types, so the backend works unchanged when it switches to `load_detector()`.

#### `load_detector(path, device="auto")`

- `path`: a bank folder (see below).
- `device`: `"cuda"`, `"cuda:N"`, `"mps"`, `"cpu"` or `"auto"`. `"auto"` uses `ARGUS_DEVICE` if it is set, otherwise the first available of CUDA, MPS and CPU.
- Raises `FileNotFoundError` if the folder or its `bank.json` is missing, and `ValueError` if `bank.json` has an unsupported `schema_version`.
- The backbone is loaded once per process and shared, so loading several banks costs one backbone plus their memory banks.

#### `Detector.score(image)`

- `image`: any PIL image, as loaded from disk or an upload. The detector converts it to RGB (MVTec AD includes greyscale categories) and applies the bank's preprocessing (resize, crop, ImageNet normalisation). Callers pass raw images.
- Returns `(score, heatmap)`:
  - `score`: a Python `float`. It means the same as in C2: the largest patch score, higher is more anomalous, flagged when `score > tau`.
  - `heatmap`: a float32 array of shape `info["input_size"]`, with the same meaning as C2's heatmaps. It is in the model's input frame, smoothed, and its maximum is usually below `score`.

#### `Detector.score_batch(images, batch_size=16)`

- Scores a list of images in chunks of `batch_size`, to bound GPU memory.
- Returns `scores` with shape `(N,)` and dtype float32, and `heatmaps` with shape `(N, H, W)` and dtype float32, in the same order as `images`.
- An empty list returns arrays of shape `(0,)` and `(0, H, W)`.
- `batch_size` changes speed only, never results.

#### `Detector.input_view(image)`

- Returns the image as the model sees it: RGB, resized and cropped like the model input, but not normalised. Its size matches the heatmap.
- The backend draws heatmap overlays on this image, not on the raw upload. With the `patchcore` preset the centre crop removes the borders, so an overlay on the raw image would be misaligned.

#### `Detector.info`

A dict with exactly these keys, read from `bank.json`:

| Key | Type | Meaning |
|---|---|---|
| `dataset`, `category`, `setting` | string | As in C2. |
| `seed` | int | As in C2. |
| `config_hash` | string | Same as the matching C2 `run.json`, so a bank can be matched with its scores and threshold. |
| `preprocess` | string | Preprocessing preset, as in C2. |
| `input_size` | [int, int] | Model input height and width; also the heatmap shape. |
| `coreset_size` | int | Patches in the memory bank. |
| `bank_mb` | float | Memory bank size in MiB, as in C2. |
| `device` | string | Device the detector runs on, in C2's format (for example `cuda:Tesla T4`). Added at load time. |

#### Bank folders

```
artifacts/banks/{dataset}/{category}/{setting}/seed{n}/
├── bank.json
└── ...            (bank data: internal to A2.3)
```

- Names follow C1 and C2, so a bank, its scores and its split share the same path below `banks/`, `scores/` and `splits/`.
- `bank.json` holds `schema_version` (`1`) and every key of `info` except `device`. It is the only file other code may read, for example so P3.2's model registry can list banks without loading them.
- Everything else in the folder (features, coreset order, file layout) belongs to A2.3 and may change without changing C3.

#### Rules

1. **Same scores as the pipeline.** For an image in a C2 run, `score_batch` on the matching bank and device returns that run's `score` within `rtol=1e-4, atol=1e-5`, and a heatmap that matches the stored float16 one within float16 precision. A2.6 scores through the same code path so this holds. Stress tests depend on it: their τ comes from the clean C2 scores.
2. **Same image, same result.** Repeated calls on one detector give the same output within the same tolerance, whatever `batch_size`.
3. **One call at a time.** A `Detector` is not thread-safe. Callers must not run two calls on the same detector at once; the backend serialises calls per detector, for example with a lock.
4. **No side effects.** Scoring never writes files and never changes the bank.

#### Example

```python
from PIL import Image

import argus

det = argus.load_detector("artifacts/banks/mvtec_ad/hazelnut/m10/seed20260801")
print(det.info["setting"], det.info["coreset_size"], det.info["input_size"])
# m10 15680 [224, 224]

img = Image.open("data/mvtec_ad/hazelnut/test/crack/004.png")
score, heatmap = det.score(img)          # float, (224, 224) float32
flagged = score > tau                    # tau from fit_threshold (C4)
overlay_base = det.input_view(img)       # 224 x 224 RGB, aligned with heatmap

images = [Image.open(p) for p in paths]
scores, heatmaps = det.score_batch(images, batch_size=32)  # (N,), (N, 224, 224)
```

### C4. Calibration interface

**Provided by:** `fit_threshold` (P1.1), the minimum-size helpers (P1.2) and `operating_point` (P1.3), all by Pratyush.
**Used by:** the results table (P2.2), stress analysis (P4.1), the backend (P3.2), and Arvind's ISP-AD analysis (A3.4) and review zone (A4.1).

C4 turns calibration scores into a cut-off τ with a false-positive guarantee, and measures FPR and recall at that cut-off. Everyone uses these functions; nobody computes a threshold or a rate any other way.

#### Interface

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


class CalibrationSizeError(ValueError):
    m: int           # calibration scores given
    m_required: int  # minimum needed for this alpha and delta


@dataclass(frozen=True)
class Threshold:
    tau: float
    rank: int                   # 1-based position of tau in the sorted calib scores
    m: int                      # number of calib scores
    alpha: float                # FPR target
    delta: float                # 1 - confidence
    achieved_confidence: float  # >= 1 - delta


@dataclass(frozen=True)
class OperatingPoint:
    fpr: float
    fpr_ci: tuple[float, float]
    recall: float
    recall_ci: tuple[float, float]
    n_normal: int
    n_defect: int
    fp: int
    tp: int


def fit_threshold(calib_scores: ArrayLike, alpha: float, delta: float = 0.05) -> Threshold: ...

def operating_point(
    scores: ArrayLike, labels: ArrayLike, tau: float, ci_level: float = 0.95
) -> OperatingPoint: ...

def min_calibration_size(alpha: float, delta: float = 0.05) -> int: ...

def min_alpha(m: int, delta: float = 0.05) -> float: ...

def save_threshold(threshold: Threshold, run_dir: str | Path) -> Path: ...

def load_threshold(run_dir: str | Path, alpha: float, delta: float = 0.05) -> Threshold: ...
```

Modules: `fit_threshold`, `Threshold`, `CalibrationSizeError`, `save_threshold` and `load_threshold` live in `argus.calibration.tolerance`; `min_calibration_size` and `min_alpha` in `argus.calibration.sample_size`; `operating_point` and `OperatingPoint` in `argus.evaluation.operating_point`.

#### Decision rule

An image is flagged as a defect if and only if `score > tau`. The comparison is strict, everywhere: in `operating_point`, the backend's verdict and the review zone.

#### `fit_threshold(calib_scores, alpha, delta=0.05)`

Distribution-free order-statistic rule (Wilks), as in Deng:

1. Sort the `m` calibration scores in ascending order, `s(1) ≤ … ≤ s(m)`.
2. The rank `k` is the smallest `k` with `P(Binomial(m, 1 − alpha) ≤ k − 1) ≥ 1 − delta`.
3. `tau = s(k)`, and `achieved_confidence = P(Binomial(m, 1 − alpha) ≤ k − 1)`.

The guarantee: if the calibration images and new good images are independent draws from the same conditions, then with probability at least `1 − delta`, the true FPR of `score > tau` is at most `alpha`.

- `calib_scores` is 1-D and every value is finite. Otherwise it raises `ValueError`.
- `alpha` and `delta` are each strictly between 0 and 1. Otherwise it raises `ValueError`.
- If `m < min_calibration_size(alpha, delta)`, no rank works, and it raises `CalibrationSizeError` with `m`, `m_required` and a message such as `need at least 59 calibration scores for alpha=0.05, delta=0.05; got 40`. P2.2 catches this to skip and warn.
- Ties need no special handling; the strict rule keeps the guarantee.

#### Minimum sizes (P1.2)

- `min_calibration_size(alpha, delta)` = `ceil(log(delta) / log(1 − alpha))`: the smallest `m` for which rank `k = m` reaches `1 − delta`.
- `min_alpha(m, delta)` = `1 − delta ** (1 / m)`: the smallest target `m` scores can support.

#### `operating_point(scores, labels, tau, ci_level=0.95)`

- `scores` and `labels` are 1-D and of equal length, `labels` holds only `0` and `1`, and every score is finite. Otherwise it raises `ValueError`.
- `n_normal` and `n_defect` are the counts of labels 0 and 1. `fp` counts label-0 rows with `score > tau`, and `tp` counts label-1 rows with `score > tau`.
- `fpr = fp / n_normal` and `recall = tp / n_defect`.
- `fpr_ci` and `recall_ci` are Wilson score intervals at `ci_level`, using `z` = the standard normal quantile at `1 − (1 − ci_level) / 2` (1.959964 for 95%). For a count `x` out of `n`, with `p = x / n`:
  - centre = `(p + z² / 2n) / (1 + z² / n)`
  - half-width = `z / (1 + z² / n) × sqrt(p(1 − p) / n + z² / 4n²)`
- If `n_normal = 0`, `fpr` and both ends of `fpr_ci` are NaN; likewise for recall when `n_defect = 0`. If both are 0, it raises `ValueError`.
- Which rows to pass is the caller's job, following C1's "How each role is used" table: for FPR, every `label = 0` row in `eval_normal` and `test`; for recall, every `label = 1` row in `test`. Passing them together in one call gives both.

#### Threshold files

- `save_threshold(t, run_dir)` writes `run_dir/thresholds/alpha{alpha}_delta{delta}.json` and returns its path.
  - `run_dir` is a clean C2 run folder.
  - Numbers in the name use Python's `:g` format, for example `alpha0.05_delta0.05.json` or `alpha0.1_delta0.05.json`.
  - It writes one file per `(alpha, delta)`, so the 10%, 5% and 2% targets of a run sit side by side.
- `load_threshold(run_dir, alpha, delta)` reads one back. It raises `FileNotFoundError` if the file is missing, and `ValueError` if the file is stale.
  - A file is stale when its `scores_sha256` no longer matches `scores.parquet`, or its `config_hash` no longer matches `run.json`.
- When A2.6 rewrites a run's scores, it deletes that run's `thresholds/` folder.
- The backend (P3.2) finds a bank's τ through its path: the bank at `banks/{dataset}/{category}/{setting}/seed{n}` uses `scores/{dataset}/{category}/{setting}/seed{n}/thresholds/`.

File contents: every `Threshold` field, plus `schema_version` (`1`), the run's `config_hash`, and `scores_sha256` (the SHA-256 of `scores.parquet` when τ was fitted). Example:

`artifacts/scores/mvtec_ad/hazelnut/m10/seed20260801/thresholds/alpha0.05_delta0.05.json`

```json
{
  "schema_version": 1,
  "tau": 2.231,
  "rank": 99,
  "m": 100,
  "alpha": 0.05,
  "delta": 0.05,
  "achieved_confidence": 0.962919,
  "config_hash": "3f9a1c07b2e4",
  "scores_sha256": "9c4e0f2a7d1b3e5f8a6c4d2b0e9f7a5c3d1b8e6f4a2c0d9e7b5a3f1c8e6d4b2a"
}
```

`tau`, `config_hash` and `scores_sha256` here are illustrative; `rank`, `m` and `achieved_confidence` are the real values for m = 100 at a 5% target.

#### Test-set lock

Code that reads `eval_normal` or `test` scores for a run first calls `load_threshold` for that run and target, and stops if it fails. `fit_threshold` only ever receives `calib` rows. This makes the plan's fair-test rule automatic: no τ can be fitted, or refitted, after looking at evaluation scores.

#### Reference values

P1.1 and P1.2's tests check these (from Deng; recomputed for this spec):

| Case | Expected |
|---|---|
| `fit_threshold`, m = 150, alpha = 0.10 | rank 142, achieved_confidence 0.9693 |
| `fit_threshold`, m = 150, alpha = 0.05 | rank 148, achieved_confidence 0.9818 |
| `fit_threshold`, m = 150, alpha = 0.02 | rank 150, achieved_confidence 0.9517 |
| `min_calibration_size`, alpha = 0.10 / 0.05 / 0.02 / 0.01 / 0.005 / 0.001 | 29 / 59 / 149 / 299 / 598 / 2995 |
| `min_alpha`, m = 150 | 0.01977 |
| `fit_threshold`, m = 40, alpha = 0.05 | `CalibrationSizeError`, `m_required = 59` |

All with delta = 0.05.

#### Example

```python
import pandas as pd

from argus.calibration.tolerance import fit_threshold, load_threshold, save_threshold
from argus.evaluation.operating_point import operating_point

run = "artifacts/scores/mvtec_ad/hazelnut/m10/seed20260801"
df = pd.read_parquet(f"{run}/scores.parquet")

t = fit_threshold(df.loc[df.role == "calib", "score"], alpha=0.05)
save_threshold(t, run)

t = load_threshold(run, alpha=0.05)  # test-set lock: only now read evaluation rows
ev = df[df.role.isin(["eval_normal", "test"])]
op = operating_point(ev.score, ev.label, t.tau)
print(f"FPR {op.fpr:.3f} {op.fpr_ci}, recall {op.recall:.3f} {op.recall_ci}")
```

The evaluation rows here are every `eval_normal` and `test` row. Because `operating_point` splits by label, that gives FPR over all held-out good images and recall over the test defects, as C1 specifies.
