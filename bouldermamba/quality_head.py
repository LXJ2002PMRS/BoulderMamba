"""Cascade-specific quality scoring, informed by Mask Scoring R-CNN (2019).

Adds ensemble disagreement and mask boundaries, whole-instance area targets,
and bounded sigmoid scores. This is an independent experimental adaptation;
quality estimation never receives GT during inference.
"""
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.roi_heads import CascadeRoIHead
from mmdet.models.utils import empty_instances
from mmdet.structures.bbox import bbox2roi


class EnsembleMaskQuality(nn.Module):
    def __init__(self, in_channels=256, width=64):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, width, 1)
        self.encode = nn.Sequential(nn.Conv2d(width+3, width, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU())
        self.hidden = nn.Linear(width*4*4+2, 128)
        self.output = nn.Linear(128, 1)
        self.reset_output()

    def reset_output(self):
        # Constant .5 factor initially preserves ranking and all detections.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, probabilities, rois):
        # Inputs are detached: additional supervision cannot disturb the
        # pretrained detector through the auxiliary branch.
        with torch.autocast(device_type='cuda', enabled=False):
            feature = self.reduce(F.adaptive_avg_pool2d(features.detach().float(), 14))
            probabilities = probabilities.detach().float()
            mean = probabilities.mean(dim=0)
            variance = probabilities.var(dim=0, unbiased=False)
            boundary = F.max_pool2d(mean, 3, 1, 1) - (-F.max_pool2d(-mean, 3, 1, 1))
            maps = F.adaptive_avg_pool2d(torch.cat((mean, variance, boundary), dim=1), 14)
            encoded = F.adaptive_avg_pool2d(self.encode(torch.cat((feature, maps), dim=1)), 4)
            boxes = rois.detach().float()[:, -4:]
            wh = (boxes[:, 2:]-boxes[:, :2]).clamp_min(1)
            geometry = torch.stack(((wh[:, 0]/wh[:, 1]).log(),
                                    (wh.prod(dim=1)/1e6).log()), dim=1).clamp(-12, 12)
            hidden = F.silu(self.hidden(torch.cat((encoded.flatten(1), geometry), dim=1)))
            return self.output(hidden).flatten()


@torch.no_grad()
def whole_instance_iou_targets(probability, mask_targets, sampling_results, gt_instances):
    ratios = []
    for sampled, instances in zip(sampling_results, gt_instances):
        masks = instances.masks
        areas = masks.areas
        for box, gt_index in zip(sampled.pos_priors.cpu().numpy(),
                                 sampled.pos_assigned_gt_inds.cpu().numpy()):
            ratio = masks[int(gt_index)].crop(box.astype('int32')).areas[0] / max(float(areas[int(gt_index)]), 1.)
            ratios.append(ratio)
    ratio = probability.new_tensor(ratios)
    predicted = probability[:, 0] >= .5
    target = mask_targets >= .5
    intersection = (predicted & target).sum(dim=(-1, -2)).float()
    gt_area = target.sum(dim=(-1, -2)).float() / ratio.clamp_min(1e-7)
    union = predicted.sum(dim=(-1, -2)).float() + gt_area - intersection
    return torch.where(ratio > 0, intersection/union.clamp_min(1),
                       torch.zeros_like(intersection)).clamp(0, 1)


@MODELS.register_module()
class QualityCascadeRoIHead(CascadeRoIHead):
    def __init__(self, quality_loss_weight=2., apply_quality=True, **kwargs):
        super().__init__(**kwargs)
        self.quality_head = EnsembleMaskQuality()
        self.quality_loss_weight = quality_loss_weight
        self.apply_quality = apply_quality

    def init_weights(self):
        super().init_weights()
        self.quality_head.reset_output()

    def _mask_forward(self, stage, x, rois):
        extractor = self.mask_roi_extractor[stage]
        features = extractor(x[:extractor.num_inputs], rois)
        return dict(mask_preds=self.mask_head[stage](features), mask_feats=features)

    def mask_loss(self, stage, x, sampling_results, batch_gt_instances):
        result = super().mask_loss(stage, x, sampling_results, batch_gt_instances)
        if stage != self.num_stages-1:
            return result
        rois = bbox2roi([s.pos_priors for s in sampling_results])
        if not len(rois):
            result['loss_mask']['loss_mask_quality'] = sum(p.sum()*0 for p in self.quality_head.parameters())
            return result
        # Match inference's ensemble on the same final-stage training RoIs.
        probabilities = []
        with torch.no_grad():
            for index in range(self.num_stages):
                logits = result['mask_preds'] if index == stage else self._mask_forward(index, x, rois)['mask_preds']
                probabilities.append(logits.detach().float().sigmoid())
        ensemble = torch.stack(probabilities)
        targets = whole_instance_iou_targets(ensemble.mean(dim=0), result['mask_targets'],
                                             sampling_results, batch_gt_instances)
        logits = self.quality_head(result['mask_feats'], ensemble, rois)
        result['loss_mask']['loss_mask_quality'] = self.quality_loss_weight * F.binary_cross_entropy_with_logits(logits, targets)
        return result

    def predict_mask(self, x, batch_img_metas, results_list, rescale=False):
        rois = bbox2roi([result.bboxes for result in results_list])
        if not len(rois):
            return empty_instances(batch_img_metas, rois.device, task_type='mask',
                                   instance_results=results_list,
                                   mask_thr_binary=self.test_cfg.mask_thr_binary)
        probabilities = []
        for stage in range(self.num_stages):
            result = self._mask_forward(stage, x, rois)
            probabilities.append(result['mask_preds'].float().sigmoid())
        ensemble = torch.stack(probabilities)
        quality = self.quality_head(result['mask_feats'], ensemble, rois).sigmoid()
        counts = [len(result) for result in results_list]
        # Official non-augmented Cascade merging is the stage mean.
        merged = ensemble.mean(dim=0).split(counts, dim=0)
        results_list = self.mask_head[-1].predict_by_feat(
            mask_preds=merged, results_list=results_list, batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg, rescale=rescale, activate_map=True)
        for result, scores in zip(results_list, quality.split(counts)):
            result.class_scores = result.scores.clone()
            result.mask_qualities = scores.to(result.scores.dtype)
            if self.apply_quality:
                result.scores = result.scores * result.mask_qualities
        return results_list
