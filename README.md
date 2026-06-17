# Vision Robustness Project

Course project. We took three different vision tasks (a low-level feature detector, a mid-level segmentation network, and a high-level object detector), beat them up with three image distortions, and then tried two ways to recover the lost performance: clean up the input image first, or fine-tune the model on the distorted data directly.

## What we picked

|                     |                                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Dataset             | [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) — validation split, 200-image subset, 150-class segmentation GT |
| Task 1 (low-level)  | Feature detection with **ORB** (`cv2.ORB_create`)                                                                                                |
| Task 2 (mid-level)  | Semantic segmentation with **SegFormer-b0** (`nvidia/segformer-b0-finetuned-ade-512-512`)                                                        |
| Task 3 (high-level) | Object detection with **YOLOv8n** (`ultralytics`)                                                                                                |
| Distortion 1        | Gaussian noise (manual `np.random.normal`, 5 levels, std ∈ {5, 10, 20, 35, 60})                                                                  |
| Distortion 2        | JPEG compression (`albumentations.ImageCompression`, quality ∈ {80, 50, 25, 10, 2})                                                              |
| Distortion 3        | Low light (`albumentations.RandomBrightnessContrast`, brightness ∈ {-0.1 … -0.9})                                                                |
| Enhancement 1       | Non-Local Means denoising — paired with Gaussian noise                                                                                           |
| Enhancement 2       | Bilateral filtering — paired with JPEG compression                                                                                               |
| Enhancement 3       | CLAHE contrast equalisation — paired with low light                                                                                              |
| Fine-tuning         | YOLOv8n retrained on noisy images using clean predictions as pseudo-labels (class-agnostic, `nc=1`)                                              |


## How we measure things

| Task                 | Metric                                                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| ORB                  | Keypoint count ratio relative to clean (1.0 means same number of keypoints as clean)                                                       |
| YOLOv8               | Detection recall against clean reference boxes — IoU ≥ 0.5, greedy one-to-one match, class-aware (or class-agnostic for the FT comparison) |
| SegFormer            | Mean IoU over 150 classes, also reported per-class for the top 20 by pixel count                                                           |
| Distortion intensity | SNR in dB — same units across all three distortions, so the curves are comparable                                                          |


## Running it

```
git clone https://github.com/katren58/image_processing_project.git
cd Image_Processing
pip install -r requirements.txt
jupyter notebook robustness_project.ipynb
```

Or open it straight in Google Colab. The first run downloads the ~900 MB ADE20K zip, the YOLO weights, and the SegFormer weights. Everything else is local. On CPU, expect 45–90 minutes; the YOLO fine-tune (cell 25) is the bottleneck.

## Repository layout

```
.
├── robustness_project.ipynb   ← main notebook (all the code lives here)
├── requirements.txt
├── README.md
└── work/                      ← gitignored except for results/
    ├── finetune_data/         ← pseudo-label dataset built from clean YOLO predictions
    ├── models/yolo_ft/        ← fine-tuned weights and ultralytics training logs
    │   └── weights/{best,last}.pt
    └── results/               ← committed: figures, JSON, npz
        ├── eda_samples.png
        ├── class_distribution.png
        ├── orb_clean.png
        ├── yolo_clean.png
        ├── segformer_clean.png
        ├── distortions_preview.png
        ├── distortion_snr_curves.png
        ├── enhancement_preview.png
        ├── per_class_iou.png             ← per-class SegFormer IoU
        ├── four_way_comparison.png
        ├── pipeline_visual.png
        ├── distortion_results.json
        ├── enhancement_results.json
        ├── seg_baseline_per_class.npz    ← per-class inter/union for the clean baseline
        ├── seg_distortion_per_class.npz  ← per-class IoU for each (distortion, level)
        ├── seg_enhancement_per_class.npz ← per-class IoU after each enhancement
        ├── top20_class_idx.npy
        └── class_counts.npy
```

## Results

All distortion and enhancement evaluations run on the first 50 images (`EVAL_N = 50`) for compute budget. Baselines (ORB and SegFormer mIoU) run on the full 200.

### Baselines on clean images

| Task      | Metric                                         | Value                                                |
| --------- | ---------------------------------------------- | ---------------------------------------------------- |
| ORB       | Mean keypoints per image                       | 473.1 ± 55.6                                         |
| YOLOv8    | Detection recall                               | 1.000 (vacuous: clean predictions are the reference) |
| YOLOv8    | Images with at least one detection             | 165 / 200                                            |
| SegFormer | Per-image mIoU (mean over 200)                 | 0.335                                                |
| SegFormer | Dataset-level mIoU (140 / 150 classes present) | 0.261                                                |


### Performance under mid-severity distortion (level 3)

| Distortion             | SNR (dB) | ORB ratio | YOLO recall | Seg mIoU |
| ---------------------- | -------- | --------- | ----------- | -------- |
| GaussNoise (std=20)    | 17.0     | 1.020     | 0.540       | 0.304    |
| JPEGCompression (q=25) | 24.8     | 1.001     | 0.583       | 0.333    |
| LowLight (-0.5)        | 2.6      | 0.908     | 0.438       | 0.277    |


The full SNR ladder (all 5 levels per distortion) is in `work/results/distortion_results.json`.

### What enhancement does at heavy distortion (level 4)

| Distortion           | Enhancer         | YOLO recall | Seg mIoU |
| -------------------- | ---------------- | ----------- | -------- |
| GaussNoise (L4)      | Non-Local Means  | 0.374       | 0.256    |
| JPEGCompression (L4) | Bilateral filter | 0.575       | 0.300    |
| LowLight (L4)        | CLAHE            | 0.176       | 0.196    |


For reference, at level 4 with no enhancement: YOLO recall is 0.391 / 0.300 / 0.203 and Seg mIoU is 0.271 / 0.283 / 0.214 (Gauss / JPEG / Low). So bilateral helped JPEG quite a lot, NLM was a wash on noise, and CLAHE actively hurt low light.

### Fine-tuning YOLOv8 on noisy images

Fine-tuned for 5 epochs on 120 GaussNoise-L3 images with the clean YOLO predictions as pseudo-labels. All four bars below use class-agnostic recall so they're directly comparable.

| Condition                            | YOLO recall (class-agnostic) |
| ------------------------------------ | ---------------------------- |
| Baseline (clean)                     | 1.000                        |
| Distorted (level 3, std=20)          | 0.544                        |
| Enhanced (NLM)                       | 0.541                        |
| **Fine-tuned (5 epochs, conf=0.01)** | **0.544**                    |


The fine-tune ended up with basically the same recall as just running standard YOLO on the noisy image. Training for more epochs would almost certainly help (see the methodology notes).

## Methodology notes & limitations

A few non-obvious choices worth flagging up front so the numbers above don't get over-interpreted.

**No bounding-box GT.** ADE20K only ships with segmentation masks, so for object detection we ran clean YOLOv8 once across all 200 images and treated those predictions as our reference set. Recall under each condition is therefore "how many of the clean detections survive", not absolute detection accuracy. The clean baseline is 1.0 by construction; only the differences between conditions are interesting.

**Fine-tune is class-agnostic.** Pseudo-labels carry no reliable class identity once you start trusting them as ground truth, so we set `nc=1, names=['object']` and trained YOLO as a single-class localizer. We're studying robustness of *where* objects are, not *what* they are. The 4-way comparison chart re-evaluates the standard YOLO with `class_agnostic=True` so the FT bar isn't unfairly advantaged.

**Why the fine-tune recall is 0.544 in this run.** Five epochs on 120 images is enough for the model to learn box regression but not enough to calibrate its objectness scores. A conf-threshold sweep on the resulting checkpoint shows ~29 detections per image at conf=0.01 but essentially zero at conf ≥ 0.02. We evaluate the FT model at conf=0.01 (the standard YOLO stays at conf=0.25 since its calibration is intact from COCO pretraining). At those settings the FT model and the un-fine-tuned model land in the same place. To get a real fine-tune win, bump cell 25 to 20–30 epochs. We left it at 5 to keep the notebook runnable on CPU.

**`detection_recall` rules.** Each ref box can match at most one prediction and vice versa: at every step we take the highest-IoU pair, count it, and remove both. Class match is required unless we explicitly say `class_agnostic=True`. Images where YOLO had no clean detections return NaN and get excluded from the mean (rather than counting as a vacuous 1.0).

**SegFormer mIoU is per-image, then averaged.** We compute mIoU for each image (averaging only over classes that appear in *that* image), then average across images. This is a different formulation from the canonical ADE20K dataset-level mIoU (`Σ inter / Σ union` over the whole set), so the number doesn't directly compare to SegFormer-b0's published 37.4. The per-class chart (`per_class_iou.png`) uses the canonical dataset-level formulation.

**GaussNoise is hand-rolled, not albumentations.** We started with `albumentations.GaussNoise(var_limit=(v,v))` and discovered all five levels collapsed to roughly 6 dB SNR (the underlying clip saturates well before our intended high std values). Replaced it with a one-line `np.random.normal(0, std, ...)` followed by `np.clip(0, 255)`, which gives a clean monotonic descent from ~28 dB down to ~8 dB.

**ORB ratios above 1.0 aren't a bug.** Noise creates spurious corners, so ORB finds *more* keypoints — not better ones. If we'd measured keypoint repeatability (matching kps clean→distorted) instead of count, the trend would invert. We kept the simpler count-ratio metric for course-project scope.

**Reproducibility caveats.** `random.seed(42)` and `np.random.seed(42)` are set at the top of the notebook, but PyTorch and CuDNN are not seeded, so the YOLO fine-tune isn't bit-reproducible across runs. The distortions themselves and per-image evaluation order are deterministic.

**50-image evaluation subset.** Distortion and enhancement evaluation is on `image_paths[:50]`. Baselines (ORB count, SegFormer mIoU) cover all 200 images. Per-class IoU is computed by accumulating intersections and unions across the 50-image subset.

## Key findings

- **JPEG compression is the gentlest of the three.** SegFormer mIoU barely moves from L1 to L3 (0.345 → 0.333) and only collapses at q=2 (mIoU 0.122). YOLO holds up well except at the most aggressive setting.
- **Gaussian noise is the worst at any given SNR.** YOLO class-strict recall slides from 0.82 at L1 (28.8 dB) down to 0.15 at L5 (8.3 dB), tracking SNR almost linearly.
- **Low light hits a cliff.** Once SNR drops below ~3 dB, ORB collapses (0.91 → 0.09 between L3 and L5) and YOLO recall drops from 0.81 to 0.01. Of the three distortions, this is the only one where ORB falls apart entirely.
- **Per-class IoU shows a clear pattern**: large smooth regions are robust, small or fine-boundary classes are fragile. Under GaussNoise L3, `sky` only loses about 2% IoU (0.951 → 0.932), but `door` loses 95% (0.328 → 0.015), `curtain` 48%, `window` 33%. The full top-20 view is in `per_class_iou.png`.
- **NLM denoising didn't recover YOLO** under noise (0.541 enhanced vs 0.544 distorted, class-agnostic). The over-smoothing washes out the same fine details YOLO needs to find objects.
- **CLAHE on low light made every metric worse.** YOLO recall went from 0.203 distorted to 0.176 enhanced; SegFormer mIoU from 0.214 to 0.196. Over-aggressive contrast pushed images outside what the networks have seen during training.
- **Bilateral filtering on JPEG was the only clear enhancement win.** YOLO recall at L4 nearly doubled (0.300 → 0.575) and SegFormer mIoU edged up too (0.283 → 0.300).
- **Fine-tuning at this budget didn't move the needle** — same recall as the un-fine-tuned model at the same evaluation threshold. With 4× the epochs we'd expect a real gain (see the methodology note).


## Visualizations

| File                        | What it shows                                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| `eda_samples.png`           | Six sample images alongside their GT segmentation masks and overlays                                      |
| `class_distribution.png`    | Top 20 ADE20K classes by pixel count — wall dominates by a wide margin                                    |
| `distortions_preview.png`   | One image distorted at all 5 severity levels, for each of the 3 distortions                               |
| `distortion_snr_curves.png` | All 3 metrics plotted against SNR, one line per distortion                                                |
| `enhancement_preview.png`   | Clean / distorted / enhanced side-by-side, one row per distortion                                         |
| `per_class_iou.png`         | Per-class SegFormer IoU for the top-20 classes — clean / distorted-L3 / enhanced-L4, panel per distortion |
| `four_way_comparison.png`   | Bar chart: clean / distorted / enhanced / fine-tuned (class-agnostic recall)                              |
| `pipeline_visual.png`       | YOLO and SegFormer outputs side-by-side across the three conditions                                       |
