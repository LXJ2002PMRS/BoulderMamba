"""FP32 mask head and RoIAlign used by the selected architecture."""
import torch
from mmdet.registry import MODELS
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor

@MODELS.register_module()
class FP32CascadeMaskHead(FCNMaskHead):
    """Preserve full pretrained heads while avoiding half-precision overflow."""

    def forward(self, features):
        with torch.autocast(device_type='cuda', enabled=False):
            return super().forward(features.float())


@MODELS.register_module()
class FP32RoIExtractor(SingleRoIExtractor):
    def forward(self,feats,rois,roi_scale_factor=None):
        # The installed MMCV CUDA kernel does not implement BF16 RoIAlign.
        # Casting preserves gradients to the BF16 backbone/FPN.
        with torch.autocast('cuda',enabled=False):
            return super().forward(tuple(f.float() for f in feats),rois.float(),roi_scale_factor)
