# Flood Segmentation with U-Net

Binary semantic segmentation of flood regions in satellite imagery using PyTorch. Given an RGB satellite image, the model predicts a binary mask at pixel level.

This project compares:
- A custom U-Net trained from scratch.
- A U-Net-style decoder with an ImageNet-pretrained ResNet-18 encoder.

Both experiments and their evaluation are documented in [`cosmo-nw.ipynb`](cosmo-nw.ipynb).

## Dataset

- **Size:** 290 satellite image–mask pairs.
- **Images:** RGB, resized to `256 × 256`.
- **Masks:** grayscale binary masks, resized to `256 × 256`.
- **Metadata:** a `metadata.csv` file with `Image` and `Mask` columns.
- **Dataset identifier used in the Kaggle notebook:** `geethchandchinni/cosmo-segmentation`.

The notebook currently uses absolute Kaggle paths. To rerun it, attach the dataset in Kaggle and ensure the `img_path`, `mask_path`, and `csv_path` variables point to the image directory, mask directory, and `metadata.csv`.

**Dataset attribution/licence:** confirm the dataset's source, licence, and redistribution conditions before adding a licence or redistributing the data. The dataset itself is not included in this repository.

## Preprocessing and augmentation

- Convert satellite images to RGB and resize to `256 × 256` using bilinear interpolation.
- Normalize images using ImageNet channel statistics.
- Convert masks to grayscale, resize with nearest-neighbor interpolation, convert to tensors, and binarize at `0.5`.
- Apply training-only horizontal flips (`p=0.5`), vertical flips (`p=0.5`), and random rotation (up to ±30°).
- Disable augmentation for validation data.
- Use a fixed random seed of `42` for the split and training setup.

The dataset is split into **232 training pairs (80%)** and **58 validation pairs (20%)**, using a shuffled split with `random_state=42`. There is **no independent test set**.

## Models

### 1. Custom U-Net baseline

- Encoder channel widths: `64, 128, 256, 512`, with a `1024`-channel bottleneck.
- Double-convolution blocks with batch normalization and ReLU.
- Decoder with transposed-convolution upsampling and skip connections.
- Dropout in encoder, bottleneck, and decoder blocks.
- **Trainable parameters:** 31,043,521 (about 31.04M).

### 2. ResNet-18 encoder U-Net

- Torchvision ResNet-18 initialized with ImageNet-pretrained weights.
- Custom U-Net-style decoder with skip connections.
- Lower learning rate for the pretrained encoder than for the decoder.
- **Trainable parameters:** 14,325,889 (about 14.33M).

The pretrained encoder is intended to reuse general visual features, while the decoder learns to produce the pixel-level flood mask.

## Training configuration

| Setting | Custom U-Net | ResNet-18 encoder U-Net |
|---|---:|---:|
| Optimizer | Adam | Adam |
| Learning rate | `1e-4` | Encoder: `1e-5`; decoder: `1e-4` |
| Batch size | 16 | 16 |
| Maximum epochs | 40 | 40 |
| Early-stopping patience | 8 epochs | 8 epochs |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Scheduler factor / patience | `0.1 / 4` | `0.1 / 4` |
| Loss | `0.5 × BCEWithLogitsLoss + 0.5 × Dice loss` | `0.5 × BCEWithLogitsLoss + 0.5 × Dice loss` |
| Checkpoint selection | Lowest validation loss | Lowest validation loss |

The recorded Kaggle run used an NVIDIA Tesla T4. The notebook saves the best checkpoints and training histories under `/kaggle/working/` when the relevant training cells are run.

## Results

All metrics below are **validation metrics**, measured on the same 58-image validation split. They are not test-set results.

### Model comparison

| Metric | Custom U-Net | ResNet-18 encoder U-Net |
|---|---:|---:|
| Trainable parameters | 31.04M | 14.33M |
| Best epoch | 31 | 35 |
| Best validation loss | 0.2485 | 0.1954 |
| Validation IoU (training-loop aggregation) | 75.35% | 79.72% |
| Validation Dice (training-loop aggregation) | 85.20% | 88.13% |
| Mean per-image IoU (separate evaluation) | 75.28% | 79.88% |
| Mean per-image Dice (separate evaluation) | 85.10% | 88.19% |

Using the training-loop metrics, the ResNet-18 encoder model improves validation IoU by **4.37 percentage points** and validation Dice by **2.93 percentage points**, while reducing the trainable parameter count by approximately **53.9%**.

The training-loop metrics and mean per-image metrics use different aggregation procedures, so compare values from the same row/method rather than treating them as interchangeable.

### Threshold analysis

The notebook evaluates thresholds from `0.20` to `0.80` in increments of `0.05` on the validation set. For the ResNet-18 encoder model:
- At threshold `0.50`, mean per-image IoU is `79.88%` and Dice is `88.19%`.
- Threshold `0.45` gives IoU `79.89%` and Dice `88.21%`.

This difference is very small, so `0.50` remains the simple default. Because threshold selection used the validation set, the tuned-threshold result is not an independent evaluation.

### Per-image analysis

The notebook also ranks individual validation images and visualizes the input image, ground-truth mask, predicted mask, and probability map. Performance varies by image: the recorded per-image IoU ranged from `21.65%` to `95.30%` for the ResNet-18 encoder model. The examples show why aggregate metrics should be supplemented with qualitative inspection.

## Metrics

- **Intersection over Union (IoU):** `TP / (TP + FP + FN)`.
- **Dice coefficient:** `2TP / (2TP + FP + FN)`.

Here, TP is a correctly predicted flood pixel, FP is a background pixel predicted as flood, and FN is a flood pixel predicted as background. Higher scores indicate better overlap between predicted and ground-truth masks.

The model outputs logits. The notebook applies sigmoid to obtain probabilities and thresholds those probabilities to form a binary mask.

## Run the notebook

The notebook was developed in Kaggle and currently contains absolute Kaggle dataset paths.

1. Open `cosmo-nw.ipynb` in Kaggle.
2. Attach the dataset identified above.
3. Check the three path variables (`img_path`, `mask_path`, and `csv_path`) against the attached dataset layout.
4. Run the cells in order.

To install the main Python dependencies in a compatible Python environment:

```bash
pip install -r requirements.txt
```

The requirements list records the libraries used by the notebook, but does not pin versions because the exact package versions from the original Kaggle session have not been recorded. For closest reproduction, use a compatible PyTorch/torchvision environment and document the versions used in any new run.

## Checkpoints and outputs

The notebook's Kaggle output names are:
- `best_flood_unet.pth` — best custom U-Net checkpoint.
- `best_pretrained_unet.pth` — best ResNet-18 encoder U-Net checkpoint.
- `training_history.csv` — custom U-Net training history.
- `pretrained_unet_history.csv` — ResNet-18 encoder U-Net training history.

These files are **not included** in this repository at present. They must be regenerated by running the relevant training cells, or deliberately added after checking file size and distribution constraints.

## Limitations

- The dataset is small (290 image–mask pairs), so results may be sensitive to the split and overfitting.
- There is no independent test set; all reported scores are from validation data used for checkpoint selection and threshold analysis.
- The split is image-level and does not establish whether related images come from the same scene or geographic region.
- Generalization to other satellite sensors, geographic regions, and flood events has not been established.
- ResNet-18 uses ImageNet-pretrained weights, not flood-specific pretraining.
- Resizing to `256 × 256` may remove fine spatial details.
- Per-image performance varies, and some scenes remain challenging.

The results demonstrate a comparison on this validation split; they should not be interpreted as independent test accuracy or as evidence of production/disaster-response readiness.
