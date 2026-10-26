"""Evaluate external COCO mask predictions; this interface needs no GPU."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from bouldermamba.evaluation import evaluate_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    metrics = evaluate_masks(args.annotations, args.predictions)
    print(json.dumps(metrics, indent=2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / ('metrics_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(metrics, stream, indent=2)
    print(output.resolve())


if __name__ == '__main__':
    main()
