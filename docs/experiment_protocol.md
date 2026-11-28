# Experiment protocol

The full protocol (seeds, FPR targets, confidence, fair-test rules) is written in task 0.5. This file currently holds the dataset decisions from task 0.4.

## Datasets, categories and split sizes (0.4)

Configs: `configs/datasets/mvtec_ad.yaml` and `configs/datasets/isp_ad.yaml`. Roles follow contract C1. Both datasets use Deng's three seeds, 20260801–20260803.

| Dataset | Category | Train good | Support | Calib | Eval normal | Test good | FPR normals | Test defects |
|---|---|---|---|---|---|---|---|---|
| MVTec AD | hazelnut | 391 | 200 | 100 | 91 | 40 | 131 | 70 |
| MVTec AD | screw | 320 | 160 | 100 | 60 | 41 | 101 | 119 |
| ISP-AD | LSM-2 | 401 | 300 | 150 | 200 | 382 ᵃ | 200 | 96 (33 point) |
| ISP-AD | LSM-1 | 3,678 | 300 | 150 | 200 | 1,470 ᵃ | 200 | 95 (46 point) |
| ISP-AD | ASM | 500 | 300 | 150 | 200 | 1,916 ᵃ | 200 | 74 (41 point) |

"FPR normals" counts every good image FPR is measured on (C1): on MVTec AD, `eval_normal` plus the original test goods; on ISP-AD, `eval_normal` only.

ᵃ ISP-AD counts are from the A1.1 download. C1 pools each modality's train and test goods before drawing the 650 split images, because LSM-2 and ASM have fewer than 650 train goods. Test goods are therefore not added to FPR normals, and pooled goods not drawn in a seed are unused. See "Downloaded counts (A1.1)" below.

**Why these MVTec AD categories.**
- **Hazelnut** has many training images (391), which leaves room to hold some out for calibration. But PatchCore already reaches 100% image AUROC on it, so memory size may change nothing visible.
- **Screw** is added because it is the clearest category where shrinking memory hurt in the ASO paper (about −0.7 AUROC). It gives the memory sweep a chance to show an effect.
- **Why not more:** the split rules, not compute, limit the choice. Each category needs 100 calibration images and at least 100 FPR normals, and most MVTec AD categories have only about 200–280 training images and 20–30 good test images. Meeting both rules would leave them a much smaller memory bank than hazelnut and screw, which would confound a study of memory size. Adding categories would mean lowering calibration to 59 images and dropping the 10% target on MVTec AD.

**Why these split sizes.**
- **Calibration:** 100 calibration images everywhere on MVTec AD. That supports the 10% and 5% FPR targets at 95% confidence, which need at least 29 and 59 images. The 2% target needs 149, which MVTec AD's training sets cannot spare, so it is run on ISP-AD only.
- **Support:** the rest of each training set is shared between support and evaluation normals. Hazelnut keeps the plan's starting point of 200 support images. Screw keeps half its training set (160) as support, since a smaller memory is exactly what hurts it.
- **The ≥ 100 rule:** both categories have at least 100 FPR normals (131 and 101), meeting the 0.4 requirement.
- **PatchCore baseline:** support is not the full training set, so AUROC on these splits may sit slightly below the papers' numbers. The paper-comparison baseline (A2.7) is run on the full training set separately.

**Precision limit.** FPR on MVTec AD is coarse at any split we can afford. At a true FPR of 5%, the 95% Wilson interval is about 8 points wide with 131 normals ([2.6%, 10.6%]) and about 9 points with 101 ([2.1%, 11.1%]). Moving more images from support to evaluation would hardly narrow this. Per-seed FPR differences between memory settings smaller than a few points therefore cannot be resolved on MVTec AD. ISP-AD, with 200 evaluation normals ([2.7%, 9.0%]) and three seeds, is the more precise test.

**ISP-AD** copies Deng's design exactly (Table 1): per modality, 300 support, 150 calibration and 200 evaluation normals, plus every real defect in `test`. This allows a direct comparison with Deng's 100%-memory results before adding 10%, 1% and ASO.
- **Normal images (counted in A1.1):** the split needs 650 per modality. `train/good` alone has 401 (LSM-2), 3,678 (LSM-1) and 500 (ASM); with `test/good` added, 783, 5,148 and 2,416.
- **Deng's defect column (resolved in A1.1):** Table 1's "point / all" is really points / area, so ASM's 33 is its area count, not a total below 41. Totals are 96, 95 and 74.
- **Comparing recall:** Deng's recall counts point defects only, while ours counts all defects. Point defects stay identifiable through `defect_type` for a like-for-like comparison.

**Sources.**
- **MVTec AD:** the standard split, as listed in published benchmark tables. A1.1 verifies the counts after download and fails loudly if they differ from `expected_counts`.
- **ISP-AD:** Deng, arXiv:2608.15090, Table 1.

## Downloaded counts (A1.1)

Commands: `make data-isp` and `make data-mvtec ARCHIVE=<path or URL>` (scripts in `scripts/`). Both lay the data out under `$ARGUS_DATA_DIR` (default `data/`), check the counts below with `python -m argus.data.verify`, and write `.argus_source.json` (archive, hash, counts, date) next to the data. Rerunning skips anything already present and verified.

### ISP-AD

Source: Zenodo record 14911043, v1.0, CC BY-NC-SA 4.0. Only `unsupervised.zip` is downloaded (425,983,226 B, MD5 `edfb54cb26629b53963fa08bec9e4a6b`, SHA-256 `79ccb202c034d972a1e73a55ec286ac1d689766b94eb0ea35f7b61f6c9f8ad3e`). The record's six supervised and additional archives (0.55–5.5 GB each) are not used.

| Modality (folder) | `train/good` | `test/good` | Normals (train + test) | `test/points` | `test/area` | Defects |
|---|---|---|---|---|---|---|
| LSM-2 (`LSM_2`) | 401 | 382 | 783 | 33 | 63 | 96 |
| LSM-1 (`LSM_1`) | 3,678 | 1,470 | 5,148 | 46 | 49 | 95 |
| ASM (`ASM`) | 500 | 1,916 | 2,416 | 41 | 33 | 74 |

Every defect image has a mask with the same filename under `ground_truth/{points,area}/`.

- **Folder names:** the archive's own names (`LSM_1`, `LSM_2`, `ASM`) are kept, and the config keys match them, because C1 ties `{category}` to the folder name and ISP-AD's own loaders expect these names. Prose keeps "LSM-1".
- **Not extracted:** `LSM_1/train/good_reduced/` (500 files, each a byte-identical copy of a `train/good` image, checked by CRC32; keeping both would let the same image land in two roles) and the archive's `configs/` folder (other methods' settings). `license.txt` and `changelog.txt` are kept in `data/isp_ad/`.
- **Deng's Table 1:** the "point / all" column (LSM-2 33/63, LSM-1 46/49, ASM 41/33) is really points / area: every "all" value equals the `area` folder count. The real defect totals are 96, 95 and 74. Deng's recall denominators in Table 2 (99 = 33×3, 138 = 46×3, 123 = 41×3) confirm the point counts. This also explains the apparent ASM inconsistency noted in 0.4.
- **Normal pool (C1):** Deng's split needs 650 normals per modality, but `train/good` has only 401 for LSM-2 and 500 for ASM. C1 therefore pools `train/good` and `test/good`, draws support, calibration and evaluation normals from that pool per seed, and keeps only defects in `test`. Pooled normals not drawn in a seed (133, 4,498 and 1,766) are unused, so FPR uses exactly 200 normals per modality, as in Deng. Using them as extra FPR normals was considered and rejected: it would break the comparison with Deng and make the modalities' precision uneven.

### MVTec AD

Source: the archive downloaded from mvtec.com after accepting the licence (CC BY-NC-SA 4.0). MVTec does not publish checksums, so the script checks file counts against `configs/datasets/mvtec_ad.yaml` and records the archive's SHA-256 so both laptops can confirm they used the same file. Only the configured categories are extracted.

| Category | `train/good` | `test/good` | Test defects | Defect types |
|---|---|---|---|---|
| hazelnut | 391 | 40 | 70 | crack 18, cut 17, hole 18, print 17 |
| screw | 320 | 41 | 119 | manipulated_front 24, scratch_head 24, scratch_neck 25, thread_side 23, thread_top 23 |

Verified on the full archive `mvtec_anomaly_detection.tar.xz` (5,264,982,680 B, SHA-256 `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`). The totals are enforced from the config. The per-type breakdown is printed but not enforced, and it matched the usual MVTec AD split exactly. Every defect image has its `ground_truth/{type}/{id}_mask.png`.

- The archive has no wrapper folder: the categories, `license.txt` and `readme.txt` sit at the top level, and each category folder also carries its own `license.txt` and `readme.txt`, which are kept.
- The archive stores folders and files as owner read-only (`r-x`, `r--`). The script makes the extracted tree owner-writable, otherwise it could not be moved into place or cleaned up.

## Split manifests (A1.2)

`make splits` writes the contract C1 manifests to `artifacts/splits/{dataset}/{category}/seed{n}.csv`, and `make check-splits` validates them. `artifacts/` is not committed, so each machine regenerates them. A machine has the same splits as everyone else when its files match the hashes below; `argus splits` prints the first 12 characters of each hash.

**Algorithm** (`argus.data.splits`). The category must first pass the A1.1 count check. Its normal pool (MVTec AD: `train/good`; ISP-AD: `train/good` plus `test/good`, per C1) is sorted by `image_id`, and `numpy.random.default_rng(seed).permutation` orders it. The first `support`, the next `calib` and the next `eval_normal` entries take those roles. The rest of the pool is unused, and every test image goes in `test`.

Generated with numpy 2.5.5 (Python 3.14). numpy guarantees the bit stream of `default_rng`'s generator but not that `Generator` methods such as `permutation` give the same output in every version (NEP 19). If the hashes differ on another machine, compare numpy versions first. `tests/test_splits.py` pins a golden manifest hash so CI catches such a change.

| Dataset | Category | Seed | Support | Calib | Eval normal | Test | SHA-256 |
|---|---|---|---|---|---|---|---|
| mvtec_ad | hazelnut | 20260801 | 200 | 100 | 91 | 110 | `b019c4449de52f2dccac24b9e7802ac341bcb2f3ef3b6ee1582640c6365606bc` |
| mvtec_ad | hazelnut | 20260802 | 200 | 100 | 91 | 110 | `fbabf50ba3e300c35049a10133a38804e364455326f3775c0c1d171fc8e3bf69` |
| mvtec_ad | hazelnut | 20260803 | 200 | 100 | 91 | 110 | `641c0bdf9d9677ac78564af75e19d4c7e738118c2612e5fdffc8814f7486349f` |
| mvtec_ad | screw | 20260801 | 160 | 100 | 60 | 160 | `bdd7d9ebc689caae5a244f7744db7ead60b7e67c9154af90e0495d00626a4194` |
| mvtec_ad | screw | 20260802 | 160 | 100 | 60 | 160 | `8646cc7aa2eba7953d9ce91bcbaeea7228e7207acfcdf38581c0c24c2a48b81a` |
| mvtec_ad | screw | 20260803 | 160 | 100 | 60 | 160 | `93c41f0b2330020a462106fe9d5f138de2e50770f23a50a3d8bd4fd0071ebc9a` |
| isp_ad | LSM_2 | 20260801 | 300 | 150 | 200 | 96 | `4778ae89fdeb5d43e88b77c4208dfe7a30a8b94aa7782f214c71a9f490a92640` |
| isp_ad | LSM_2 | 20260802 | 300 | 150 | 200 | 96 | `4637f94f3e1c087ee15aca85573cc6df372b8e0f91a154f42ea83cd19ae2b15a` |
| isp_ad | LSM_2 | 20260803 | 300 | 150 | 200 | 96 | `8ba817c55f25a325ceff44b0fb5a807b1e5fdacdef7cc94e8e1d711a2b97b976` |
| isp_ad | LSM_1 | 20260801 | 300 | 150 | 200 | 95 | `dac6b65fecc43b0e9d93a267da62c00822650c6a75449afe1d008d103cd86f89` |
| isp_ad | LSM_1 | 20260802 | 300 | 150 | 200 | 95 | `c34e5d6804e8d7d087aa759a0a1851c5972f34daaaceb2221b8eb733e8ebfe95` |
| isp_ad | LSM_1 | 20260803 | 300 | 150 | 200 | 95 | `78f7bb41438571cf9c947d09fedb4c4337aa20787d78c7061ef975c726d766da` |
| isp_ad | ASM | 20260801 | 300 | 150 | 200 | 74 | `de08678eef4a16f0a134619c08ca85e072ac1c7beeae960576cc7510fe900a5a` |
| isp_ad | ASM | 20260802 | 300 | 150 | 200 | 74 | `45a25a00d9db09735b42814efbb758ac9f4046ba5c7e9c35a2f38b0ea0bdccae` |
| isp_ad | ASM | 20260803 | 300 | 150 | 200 | 74 | `090740b7878d7321938157b391c06c227f9fb446c3d3fc600ae638a9154f5eb0` |

`test` is identical across seeds: MVTec AD's whole original test set (hazelnut 40 good + 70 defect, screw 41 + 119), and ISP-AD's defects only (points + area). Because ISP-AD draws from the pooled goods, a large share of its 650 split images come from `test/good`: about 80% for ASM, 50% for LSM-2 and 30% for LSM-1, in proportion to the pool. Every role is drawn from the same pool, so calibration and evaluation normals stay exchangeable, which is what the Wilks threshold needs.
