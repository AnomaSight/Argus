# Experiment protocol

The full protocol (seeds, FPR targets, confidence, fair-test rules) is written in task 0.5. This file currently holds the dataset decisions from task 0.4.

## Datasets, categories and split sizes (0.4)

Configs: `configs/datasets/mvtec_ad.yaml` and `configs/datasets/isp_ad.yaml`. Roles follow contract C1. Both datasets use Deng's three seeds, 20260801–20260803.

| Dataset | Category | Train good | Support | Calib | Eval normal | Test good | FPR normals | Test defects |
|---|---|---|---|---|---|---|---|---|
| MVTec AD | hazelnut | 391 | 200 | 100 | 91 | 40 | 131 | 70 |
| MVTec AD | screw | 320 | 160 | 100 | 60 | 41 | 101 | 119 |
| ISP-AD | LSM-2 | ≥ 650 | 300 | 150 | 200 | — | 200 | 63 (33 point) |
| ISP-AD | LSM-1 | ≥ 650 | 300 | 150 | 200 | — | 200 | 49 (46 point) |
| ISP-AD | ASM | ≥ 650 | 300 | 150 | 200 | — | 200 | ? (41 point) |

"FPR normals" counts every good image FPR is measured on: `eval_normal` plus the original test goods (C1).

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
- **Points for A1.1:** the number of normal images per modality is not published. The split needs at least 650 per modality, and A1.1 records the real count.
- **ASM inconsistency:** Deng's table lists 33 defects in total for ASM, but also 41 point defects, which should be a subset. A1.1 counts the real number.
- **Comparing recall:** Deng's recall counts point defects only, while ours counts all defects. Point defects stay identifiable through `defect_type` for a like-for-like comparison.

**Sources.**
- **MVTec AD:** the standard split, as listed in published benchmark tables. A1.1 verifies the counts after download and fails loudly if they differ from `expected_counts`.
- **ISP-AD:** Deng, arXiv:2608.15090, Table 1.
