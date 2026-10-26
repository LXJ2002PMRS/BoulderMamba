"""BoulderMamba network architecture and native-pixel inference."""
from .build import build_model
from .inference import predict_image

__all__ = ['build_model', 'predict_image']
