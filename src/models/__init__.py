"""Model registry."""

from src.models.dense_world_model import DenseWorldModel
from src.models.object_world_model import ObjectWorldModel
from src.models.pixel_predictor import PixelPredictor

__all__ = ["PixelPredictor", "DenseWorldModel", "ObjectWorldModel"]
