"""Predict one image or all images in an external COCO image list."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from bouldermamba import build_model, predict_image
from bouldermamba.evaluation import instances_to_coco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--image', type=Path)
    inputs.add_argument('--annotations', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--image-id', type=int, default=0)
    parser.add_argument('--category-id', type=int, default=1)
    parser.add_argument('--backbone-root', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    if args.annotations and args.data_root is None:
        parser.error('--annotations requires --data-root.')
    if args.image:
        images = [dict(id=args.image_id, path=args.image)]
        category_id = args.category_id
    else:
        listing = json.loads(args.annotations.read_text(encoding='utf-8'))
        if len(listing['categories']) != 1:
            raise ValueError('Expected one foreground category.')
        images = [dict(id=im['id'], path=args.data_root / im['file_name']) for im in listing['images']]
        category_id = listing['categories'][0]['id']
    model = build_model(args.checkpoint, args.backbone_root, args.device)
    rows = []
    for index, image in enumerate(images, 1):
        result = predict_image(model, image['path'])
        rows.extend(instances_to_coco(result, image['id'], category_id))
        print('image %d/%d' % (index, len(images)), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / ('prediction_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(rows, stream)
    print(output.resolve())


if __name__ == '__main__':
    main()
