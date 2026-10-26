"""Single-image inference with the selected native-pixel preprocessing."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .runtime import require_cuda


@torch.inference_mode()
def predict_image(model, image_path):
    """Return MMDetection InstanceData for one native 500 x 500 image.

    Boxes use native xyxy coordinates; masks retain the native 500 x 500 grid.
    The model uses image content only and receives no labels or annotations.
    """
    from mmdet.structures import DetDataSample

    device = require_cuda(next(model.parameters()).device)
    image_path = Path(image_path).resolve()
    with Image.open(image_path) as image:
        if image.size != (500, 500):
            raise ValueError('Expected a native 500 x 500 image; tile larger images before inference.')
        array = np.array(image.convert('RGB'), copy=True)
    tensor = torch.from_numpy(array).to(device).permute(2, 0, 1).flip(0).float()
    tensor = F.interpolate(tensor[None], size=(1500, 1500), mode='bilinear',
                           align_corners=False).round().clamp(0, 255)
    prep = model.data_preprocessor
    if prep._channel_conversion:
        tensor = tensor[:, [2, 1, 0]]
    tensor = F.pad((tensor - prep.mean) / prep.std, (0, 4, 0, 4))
    sample = DetDataSample(metainfo=dict(
        img_id=0, img_path=str(image_path), ori_shape=(500, 500),
        img_shape=(1500, 1500), pad_shape=(1504, 1504),
        batch_input_shape=(1504, 1504), scale_factor=(3., 3.), flip=False))
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model.predict(tensor, [sample], rescale=True)[0].pred_instances
