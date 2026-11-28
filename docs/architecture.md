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
