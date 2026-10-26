"""Positive proposal sampling retained for the training forward interface."""
import torch
from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.samplers import RandomSampler

@TASK_UTILS.register_module()
class InstanceBalancedSampler(RandomSampler):
    """Spend positive slots on distinct GTs before filling extra candidate slots."""

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        positive = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(1)
        return self._balanced_pool(assign_result, positive, num_expected)

    def _balanced_pool(self, assign_result, positive, num_expected):
        if num_expected <= 0:
            return positive[:0]
        if positive.numel() <= num_expected:
            return positive
        gt_ids = assign_result.gt_inds[positive]
        unique = gt_ids.unique()
        unique = unique[torch.randperm(len(unique), device=unique.device)]
        chosen = []
        for gt_id in unique[:num_expected]:
            candidates = positive[gt_ids == gt_id]
            chosen.append(candidates[torch.randint(len(candidates), (), device=candidates.device)])
        selected = torch.stack(chosen)
        remaining = num_expected - len(selected)
        if remaining:
            keep = torch.ones(assign_result.gt_inds.numel(), dtype=torch.bool, device=positive.device)
            keep[selected] = False
            pool = positive[keep[positive]]
            selected = torch.cat((selected, self.random_choice(pool, remaining)))
        return selected


@TASK_UTILS.register_module()
class ProposalFirstInstanceSampler(InstanceBalancedSampler):
    """Prioritize predicted proposals and cap exact-GT positive backfill."""

    def __init__(self, max_gt_positive_fraction=0.25, **kwargs):
        super().__init__(**kwargs)
        self.max_gt_positive_fraction = max_gt_positive_fraction

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        positive = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(1)
        if not self.add_gt_as_proposals:
            return self._balanced_pool(assign_result, positive, num_expected)
        real = positive[positive >= assign_result.num_gts]
        selected = self._balanced_pool(assign_result, real, num_expected)
        remaining = min(num_expected - len(selected), int(num_expected * self.max_gt_positive_fraction))
        if remaining > 0:
            synthetic = positive[positive < assign_result.num_gts]
            if len(selected):
                covered = assign_result.gt_inds[selected].unique()
                synthetic = synthetic[~torch.isin(assign_result.gt_inds[synthetic], covered)]
            extra = self._balanced_pool(assign_result, synthetic, remaining)
            selected = torch.cat([selected, extra])
        return selected
