"""Train BoulderMamba from COCO masks; select checkpoints on validation only."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from bouldermamba import build_model, predict_image
from bouldermamba.evaluation import evaluate_masks, instances_to_coco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--train-ann', required=True, type=Path)
    parser.add_argument('--val-ann', required=True, type=Path)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--backbone-root', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--accumulation-steps', type=int, default=config.ACCUMULATION_STEPS)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output-root', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    if min(args.epochs, args.accumulation_steps) < 1 or args.lr <= 0 or args.workers < 0:
        parser.error('Epochs, accumulation and learning rate must be positive; workers must be nonnegative.')
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    model = build_model(args.init_checkpoint, args.backbone_root, args.device)
    if args.init_checkpoint is None:
        model.init_weights()

    from mmengine.dataset import pseudo_collate
    from mmdet.registry import DATASETS
    train_path = args.train_ann.resolve()
    val_path = args.val_ann.resolve()
    training = json.loads(train_path.read_text(encoding='utf-8'))
    validation = json.loads(val_path.read_text(encoding='utf-8'))
    if len(training['categories']) != 1 or len(validation['categories']) != 1:
        raise ValueError('This architecture requires one foreground category.')
    for record in training['images'] + validation['images']:
        if (record['width'], record['height']) != (500, 500):
            raise ValueError('All training/validation images must be native 500 x 500.')
    train_files = {(args.data_root / im['file_name']).resolve() for im in training['images']}
    val_files = {(args.data_root / im['file_name']).resolve() for im in validation['images']}
    if train_files & val_files:
        raise ValueError('Training and validation images must be disjoint.')
    dataset = DATASETS.build(dict(
        type='CocoDataset', data_root=str(args.data_root.resolve()), ann_file=str(train_path),
        data_prefix=dict(img=''), metainfo=dict(classes=(training['categories'][0]['name'],)),
        filter_cfg=dict(filter_empty_gt=False), pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', scale=(1500, 1500), keep_ratio=True),
            dict(type='RandomFlip', prob=.75, direction=['horizontal', 'vertical', 'diagonal']),
            dict(type='PackDetInputs')]))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.workers,
                        collate_fn=pseudo_collate, pin_memory=True)
    if not len(loader) or not validation['images']:
        raise ValueError('Training and validation must both contain images.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    run = args.output_root / ('train_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True, exist_ok=False)
    best_ap = -1.
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for index, data in enumerate(loader):
            prepared = model.data_preprocessor(data, training=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = model.loss(prepared['inputs'], prepared['data_samples'])
                loss, _ = model.parse_losses(losses)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss; checkpoint not selected.')
            window_start = (index // args.accumulation_steps) * args.accumulation_steps
            window_length = min(args.accumulation_steps, len(loader) - window_start)
            (loss / window_length).backward()
            if (index + 1) % args.accumulation_steps == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP,
                                              error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())
            print('epoch %d/%d batch %d/%d loss %.6f' %
                  (epoch + 1, args.epochs, index + 1, len(loader), float(loss.detach())), flush=True)
        model.eval()
        rows = []
        category_id = validation['categories'][0]['id']
        for image in validation['images']:
            result = predict_image(model, args.data_root / image['file_name'])
            rows.extend(instances_to_coco(result, image['id'], category_id))
        metrics = evaluate_masks(val_path, rows)
        record = dict(epoch=epoch + 1, training_loss=total_loss / len(loader), validation=metrics)
        with (run / 'training_log.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        checkpoint = dict(state_dict=model.state_dict(), meta=dict(
            model='BoulderMamba', epoch=epoch + 1, validation_mask_ap=metrics['mask_ap'],
            train_annotations=str(train_path), validation_annotations=str(val_path)))
        with (run / ('epoch_%03d.pth' % (epoch + 1))).open('wb') as stream:
            torch.save(checkpoint, stream)
        if metrics['mask_ap'] is not None and metrics['mask_ap'] > best_ap:
            best_ap = metrics['mask_ap']
            with (run / 'best_model.pth').open('wb') as stream:
                torch.save(checkpoint, stream)
        print(json.dumps(record), flush=True)
    print(run.resolve())


if __name__ == '__main__':
    main()
