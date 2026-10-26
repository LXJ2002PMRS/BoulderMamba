"""Construct the complete model and strictly load an optional checkpoint."""
from pathlib import Path
import runpy

import torch

from .runtime import numpy_checkpoint_compatibility, register_modules, require_cuda


def build_model(checkpoint=None, backbone_root=None, device='cuda:0'):
    """Build the model on CUDA. Omit checkpoint only for architecture inspection.

    Checkpoints must be trusted PyTorch files supplied by the caller. Both the
    original source/refiner bundle and a complete state_dict are accepted.
    Loading is strict: incomplete or incompatible weights raise an exception.
    """
    device = require_cuda(device)
    register_modules(backbone_root)
    from mmengine.config import Config
    from mmdet.registry import MODELS

    config_path = Path(__file__).resolve().parents[1] / 'configs/bouldermamba.py'
    config = Config(dict(model=runpy.run_path(str(config_path))['model']))
    model = MODELS.build(config.model).to(device)
    if checkpoint is not None:
        numpy_checkpoint_compatibility()
        bundle = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
        if 'source_state_dict' in bundle:
            state = dict(bundle['source_state_dict'])
            for name, value in bundle['refiner_state_dict'].items():
                state['roi_head.native_refiner.' + name] = value
        else:
            state = bundle.get('state_dict', bundle)
        model.load_state_dict(state, strict=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return model.eval()
