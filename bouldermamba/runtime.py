"""Register project modules and an explicitly located upstream backbone."""
import importlib
import os
import sys
from pathlib import Path

import torch

_REGISTERED_BACKBONE = None


def require_cuda(device='cuda:0'):
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('BoulderMamba requires CUDA; CPU inference is unavailable.')
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('The supplied BF16 inference path requires an Ampere or newer GPU.')
    return device


def register_modules(backbone_root=None):
    """Load the official detection backbone without copying it into this package."""
    global _REGISTERED_BACKBONE
    root = Path(backbone_root or os.environ.get(
        'MAMBAVISION_ROOT', Path(__file__).resolve().parents[1] / 'third_party/MambaVision'))
    source = (root / 'object_detection/tools/mamba_vision.py').resolve()
    if not source.is_file():
        raise FileNotFoundError(
            'Set MAMBAVISION_ROOT to an official MambaVision checkout containing '
            'object_detection/tools/mamba_vision.py. See README.md.')
    if _REGISTERED_BACKBONE is not None:
        if _REGISTERED_BACKBONE != source:
            raise RuntimeError('Use a fresh Python process when changing MambaVision checkout.')
        return

    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    from .selective_scan import install
    install()
    sys.path.insert(0, str(source.parent))
    existing = sys.modules.get('mamba_vision')
    if existing is not None and Path(existing.__file__).resolve() != source:
        raise RuntimeError('Another mamba_vision module is already imported; use a fresh process.')
    importlib.import_module('mamba_vision')
    for name in ('precision', 'samplers', 'box_head', 'quality_head', 'detector'):
        importlib.import_module('bouldermamba.' + name)
    from .cuda_ops import install_cuda_bf16_interpolate
    install_cuda_bf16_interpolate()
    _REGISTERED_BACKBONE = source


def numpy_checkpoint_compatibility():
    """Allow NumPy 2 scalar metadata to be read with the supplied NumPy 1 stack."""
    import numpy as np
    if int(np.__version__.split('.')[0]) >= 2:
        return
    sys.modules.setdefault('numpy._core', np.core)
    for name in ('multiarray', 'numeric', 'umath', '_multiarray_umath'):
        sys.modules.setdefault('numpy._core.' + name, importlib.import_module('numpy.core.' + name))
