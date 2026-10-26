"""COCO mask evaluation using the paper's existing detection budget."""
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


def instances_to_coco(instances, image_id, category_id=1):
    instances = instances.cpu()
    rows = []
    for index in range(len(instances)):
        mask = instances.masks[index].numpy().astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('ascii')
        x0, y0, x1, y1 = instances.bboxes[index].tolist()
        rows.append(dict(image_id=int(image_id), category_id=int(category_id),
                         score=float(instances.scores[index]),
                         bbox=[x0, y0, x1 - x0, y1 - y0], segmentation=rle))
    return rows


def evaluate_masks(annotations, predictions):
    """Evaluate a COCO result list. Returned metrics are fractions in [0, 1].

    Mask area determines size groups. Empty predictions remain valid inputs.
    Metric computation is CPU-only and does not perform network inference.
    """
    if isinstance(predictions, (str, Path)):
        predictions = json.loads(Path(predictions).read_text(encoding='utf-8'))
    if not isinstance(predictions, list):
        raise TypeError('Predictions must be a COCO result list.')
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO(str(annotations))
        if not truth.imgs or not truth.getAnnIds():
            raise ValueError('Evaluation annotations must contain images and instances.')
        known_images = set(truth.imgs)
        known_categories = set(truth.cats)
        if any(row['image_id'] not in known_images or row['category_id'] not in known_categories
               for row in predictions):
            raise ValueError('Prediction image/category IDs must match the ground-truth COCO file.')
        # Excluding bbox makes pycocotools use mask area for size groups.
        masks = [{k: row[k] for k in ('image_id', 'category_id', 'score', 'segmentation')}
                 for row in predictions]
        if masks:
            detected = truth.loadRes(masks)
        else:
            detected = COCO()
            detected.dataset = dict(images=list(truth.imgs.values()),
                                    categories=list(truth.cats.values()), annotations=[])
            detected.createIndex()
        evaluator = COCOeval(truth, detected, 'segm')
        evaluator.params.maxDets = [1, 10, 550]
        evaluator.evaluate()
        evaluator.accumulate()

    precision = evaluator.eval['precision']
    recall = evaluator.eval['recall']
    def mean_valid(values):
        values = values[values > -1]
        return float(values.mean()) if values.size else None
    # COCOeval.summarize hardcodes maxDets=100 for its first AP statistic.
    # Read the evaluated tensors directly for the selected maxDets=550.
    return dict(
        mask_ap=mean_valid(precision[:, :, :, 0, -1]),
        ap50=mean_valid(precision[np.isclose(evaluator.params.iouThrs, .50), :, :, 0, -1]),
        ap75=mean_valid(precision[np.isclose(evaluator.params.iouThrs, .75), :, :, 0, -1]),
        aps=mean_valid(precision[:, :, :, 1, -1]),
        apm=mean_valid(precision[:, :, :, 2, -1]),
        apl=mean_valid(precision[:, :, :, 3, -1]),
        ar550=mean_valid(recall[:, :, 0, -1]))
