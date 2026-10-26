# BoulderMamba

BoulderMamba performs asteroid boulder instance segmentation using MambaVision-T, a three-stage Cascade detector, distributional residual box refinement, mask-quality scoring and native-pixel refinement (NPR).

This repository contains the network architecture and training, prediction and evaluation interfaces. Data, weights, predictions, experiment results and ablation workflows are not included.

## Structure

```text
bouldermamba/             # Detector, box/quality heads, NPR, losses and CUDA utilities
configs/bouldermamba.py   # Complete network definition
config.py                # Editable starting settings for training
train.py                 # COCO-mask training; validation-selected checkpoints
predict.py               # Single-image or COCO-image-list prediction
eval.py                  # COCO mask evaluation
```

NPR uses a 96 × 96 integer-aligned patch on the original pixel grid. Confidence is an input descriptor and never gates correction or supervision. Pixels outside the patch retain the coarse mask.

## Installation

Inference and training require an NVIDIA CUDA GPU with BF16 support (Ampere or newer). The reference stack is Python 3.8, PyTorch 2.0.1/CUDA 11.8, MMCV 2.0.0 and MMDetection 3.3.0.

```bash
conda create -n bouldermamba python=3.8.20 -y
conda activate bouldermamba
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
python -m pip install -r requirements.txt
git clone https://github.com/NVlabs/MambaVision.git third_party/MambaVision
```

Run these commands from this repository. MMCV must include compiled CUDA operators; `mmcv-lite` is insufficient. If a matching wheel is unavailable, follow the [MMCV build instructions](https://mmcv.readthedocs.io/en/2.x/get_started/build.html). The optional `mamba_ssm` package is not required: an included CUDA PyTorch scan is used when it is absent.

The backbone loads from the official `object_detection/tools/mamba_vision.py`. To use another checkout location, set `MAMBAVISION_ROOT` or pass `--backbone-root /path/to/MambaVision` to training or prediction.

## Training

Provide native **500 × 500** images and COCO instance-mask annotations with one foreground category. Image `file_name` entries are relative to `--data-root`; annotation arguments are file paths.

```bash
python train.py --data-root /path/to/images --train-ann /path/to/train.json --val-ann /path/to/validation.json
```

Use `--init-checkpoint /path/to/model.pth` to initialize from a compatible complete checkpoint. Without it, training starts from random weights. Adjust `config.py` or use `--epochs`, `--lr` and `--accumulation-steps`. These are starting settings for this interface; they do not reconstruct every historical optimization stage of the paper model.

Each run creates a separate dated `outputs/train_*/` directory. Checkpoints are selected by validation mask AP. The training interface accepts no test split. It saves model weights and logs; restarting the optimizer from a previous run is not implemented.

## Prediction

```bash
python predict.py --checkpoint /path/to/model.pth --image /path/to/image.png
python predict.py --checkpoint /path/to/model.pth --data-root /path/to/images --annotations /path/to/test.json
```

The COCO-list mode reads image metadata and category IDs; annotation masks never enter the network. For a single image, use `--image-id` and `--category-id` when IDs must match a COCO evaluation file.

Images are enlarged to 1500 × 1500 with bilinear interpolation and padded to 1504 × 1504 for the detector. Output boxes and masks use the native 500 × 500 grid. Predictions are written as a COCO result list to a dated JSON file under `outputs/`.

```python
from bouldermamba import build_model, predict_image

model = build_model(checkpoint="/path/to/model.pth")
instances = predict_image(model, "/path/to/image.png")
```

Checkpoint loading is strict. Both the original `source_state_dict` / `refiner_state_dict` bundle and a complete `state_dict` are supported. Omitting a checkpoint constructs an untrained model for inspection.

## Evaluation

```bash
python eval.py --annotations /path/to/test.json --predictions outputs/prediction_DATE.json
```

Evaluation uses COCO mask IoU 0.50–0.95 with `maxDets=[1,10,550]` and mask-based size groups. It prints AP, AP50, AP75, size-specific AP and AR550 as fractions, and saves a dated metrics JSON. This interface can run on CPU. Generated outputs and local inputs are excluded by `.gitignore`.

## License and acknowledgments

Original project code uses the [MIT License](LICENSE). Dependencies retain their own licenses. [MambaVision](https://github.com/NVlabs/MambaVision) uses the [NVIDIA Source Code License-NC](https://github.com/NVlabs/MambaVision/blob/main/LICENSE), which restricts its use to non-commercial research or evaluation. The backbone is obtained separately; MIT does not replace NVIDIA's terms.

The network builds on [MMDetection](https://github.com/open-mmlab/mmdetection), [MMCV](https://github.com/open-mmlab/mmcv), MambaVision, Cascade R-CNN and Mask Scoring R-CNN. Distributional residual refinement is informed by [D-FINE](https://github.com/Peterande/D-FINE) and adapted to Cascade box heads.
