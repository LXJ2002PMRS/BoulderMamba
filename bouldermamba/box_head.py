"""Distributional residual box refinement for the Cascade heads.

Informed by D-FINE; this is a Cascade adaptation, not its DETR decoder.
"""
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead

def interpolated_distribution_loss(logits, targets, support):
    targets = targets.detach().float().clamp(float(support[0]),float(support[-1]))
    upper = torch.searchsorted(support, targets.contiguous(), right=True).clamp(1,len(support)-1)
    lower = upper-1
    alpha = (targets-support[lower])/(support[upper]-support[lower])
    logits = logits.reshape(-1,len(support)).float()
    return ((1-alpha.flatten())*F.cross_entropy(logits,lower.flatten(),reduction='none')+
            alpha.flatten()*F.cross_entropy(logits,upper.flatten(),reduction='none')).mean()


@MODELS.register_module()
class ResidualDistributionBBoxHead(ConvFCBBoxHead):
    def __init__(self, distribution_bins=17, distribution_range=4., distribution_loss_weight=.2, **kwargs):
        super().__init__(**kwargs)
        if not (self.reg_class_agnostic or self.num_classes == 1) or not self.reg_decoded_bbox:
            raise ValueError('This adaptation requires the audited four-coordinate decoded-GIoU head')
        if distribution_bins < 3 or distribution_bins % 2 != 1 or distribution_range <= 0:
            raise ValueError('Symmetric residual bins require an odd bin count and positive range')
        self.distribution_bins = distribution_bins
        self.distribution_loss_weight = distribution_loss_weight
        grid = torch.linspace(-1,1,distribution_bins)
        self.register_buffer('distribution_support',distribution_range*grid.sign()*grid.square(),persistent=False)
        self.fc_distribution = nn.Linear(self.fc_reg.in_features,4*distribution_bins)
        self.fc_reg.register_forward_pre_hook(self._capture_reg_features)
        self.reset_distribution()

    def reset_distribution(self):
        nn.init.zeros_(self.fc_distribution.weight)
        nn.init.zeros_(self.fc_distribution.bias)

    def init_weights(self):
        super().init_weights()
        self.reset_distribution()

    def _capture_reg_features(self, module, inputs):
        self._reg_features = inputs[0].detach()

    def forward(self, x):
        cls_score, base = super().forward(x)
        logits = self.fc_distribution(self._reg_features).reshape(-1,4,self.distribution_bins)
        probability = logits.float().softmax(dim=-1)
        # Subtracting the uniform mean makes initialization exactly zero,
        # rather than relying on a near-zero floating-point symmetric sum.
        residual = ((probability-1/self.distribution_bins)*self.distribution_support).sum(dim=-1)
        self._distribution_logits = logits
        self._base_prediction = base.detach()
        return cls_score, base+residual.to(base.dtype)

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights,
             bbox_targets, bbox_weights, reduction_override=None):
        result = super().loss(cls_score,bbox_pred,rois,labels,label_weights,
                              bbox_targets,bbox_weights,reduction_override=reduction_override)
        positive = (labels>=0)&(labels<self.num_classes)
        if positive.any():
            # With reg_decoded_bbox=True, MMDetection supplies absolute GT
            # corners; convert back to the original stage-specific coder.
            encoded = self.bbox_coder.encode(rois[positive,1:],bbox_targets[positive])
            targets = encoded-self._base_prediction[positive].float()
            result['loss_distribution'] = self.distribution_loss_weight*interpolated_distribution_loss(
                self._distribution_logits[positive],targets,self.distribution_support)
        else:
            result['loss_distribution'] = self._distribution_logits.sum()*0
        return result
