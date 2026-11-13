# FORESTesprit — Dead Tree Segmentation from Aerial Imagery

Detecting and counting dead / stressed trees in aerial RGB images of US forests, using deep-learning segmentation.
This repository holds the original research notebooks, the dataset and the trained U-Net models.
The cleaned-up, production version of the project (scripts, tests, Streamlit app, Docker) lives in the
[`forest-tree-detection`](https://github.com/maryembrik/forest-tree-detection) submodule.

---

## What it does

1. **Loads** 444 aerial RGB patches and their binary masks (dead tree = white, background = black), resized to 224×224.
2. **Augments** them with Albumentations (flips, rotation, brightness/contrast, shift-scale-rotate).
3. **Trains a U-Net** (4-level encoder/decoder, 64→1024 filters, sigmoid output) with **Dice loss**, Adam (lr 1e-4), early stopping.
4. **Evaluates** with Dice, IoU, precision, recall and pixel accuracy.
5. **Counts dead trees** per image by finding connected components in the predicted masks, exports `bad_tree_counts.csv` and draws annotated masks with contours.
6. **Compares U-Net with Mask R-CNN** (torchvision ResNet50-FPN, COCO pre-trained) on Dice / IoU and overlays the predictions on the RGB images.

```
RGB patch (224×224) ──► augmentation ──► U-Net ──► binary mask ──► connected components ──► dead-tree count
                                    └──► Mask R-CNN ──► instance masks ──┘        (compared via Dice / IoU)
```

---

## Results (U-Net)

| Split | Dice | IoU | Precision | Recall | Pixel accuracy |
|-------|:----:|:---:|:---------:|:------:|:--------------:|
| Validation | 0.52 | 0.35 | 0.51 | 0.53 | 98.1 % |
| Test | 0.48 | — | — | — | 98.6 % |

- Pixel accuracy is high because most pixels are background; **Dice/IoU are the meaningful metrics**.
- About 1,300 dead trees were counted across the 52 validation images.
- The notebook's U-Net vs Mask R-CNN table (U-Net Dice 0.03 vs 0.22) is **misleading**: U-Net masks were saved as
  `unet_mask_1.png, …` so they never matched the ground-truth file names. The real U-Net Dice is ≈ 0.55.
  The bug and the fix are explained in
  [`forest-tree-detection/docs/RESULTS.md`](https://github.com/maryembrik/forest-tree-detection/blob/master/docs/RESULTS.md).

---

## Repository layout

```
FORESTesprit/
├── forestEsprit.ipynb            # Original notebook (Google Colab, Google Drive paths)
├── forestEsprit_jupyter.ipynb    # Same pipeline adapted to run locally in Jupyter (relative paths)
├── config.json, metadata.json    # Keras model architecture / metadata (Keras 3.8)
├── USA_segmentation/
│   ├── RGB_images/               # 444 aerial RGB patches  (RGB_<tile>.png)
│   ├── masks/                    # 445 binary ground-truth masks (mask_<tile>.png)
│   ├── best_unet_model.h5        # Best U-Net checkpoint (Colab run)      — Git LFS
│   ├── model.weights.h5          # U-Net weights                          — Git LFS
│   └── models/best_unet_model.h5 # Best U-Net checkpoint (local run)      — Git LFS
└── forest-tree-detection/        # Submodule: refactored project (src/, app/, tests/, Docker)
```

---

## Getting started

```bash
# Git LFS is needed for the .h5 / .keras model files
git lfs install

# --recursive also fetches the forest-tree-detection submodule
git clone --recursive https://github.com/maryembrik/FORESTesprit.git
cd FORESTesprit

pip install tensorflow albumentations opencv-python scikit-learn scikit-image matplotlib pandas tqdm torch torchvision
jupyter notebook forestEsprit_jupyter.ipynb
```

In the first cell of `forestEsprit_jupyter.ipynb`, set `BASE_DIR` to the `USA_segmentation` folder of your clone.

To load the trained model without retraining:

```python
from tensorflow.keras.models import load_model
model = load_model("USA_segmentation/models/best_unet_model.h5",
                   custom_objects={"dice_coefficient": dice_coefficient, "dice_loss": dice_loss})
```

(`dice_coefficient` and `dice_loss` are defined in the notebook.)

---

## Limitations & next steps

- Dice ≈ 0.5 leaves room for improvement. Ideas: a pre-trained encoder (EfficientNet / ResNet), combined Dice + Focal loss, attention gates, more data.
- Mask R-CNN was used with COCO weights and was not fine-tuned on tree data, so the comparison is only a baseline.
- Connected-component counting merges touching trees; watershed or instance segmentation would separate them better.

These improvements are explored in [`forest-tree-detection`](https://github.com/maryembrik/forest-tree-detection).

---

## Tech stack

Python · TensorFlow / Keras · PyTorch / torchvision · Albumentations · OpenCV · scikit-learn · scikit-image · Google Colab / Jupyter
