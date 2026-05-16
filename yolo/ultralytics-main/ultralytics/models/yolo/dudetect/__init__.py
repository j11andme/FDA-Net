# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# from .predict import DetectionPredictor
from .train import DuDetectionTrainer
from .val import DuDetectionValidator
__all__ = "DuDetectionTrainer", "DuDetectionValidator", 
# __all__ = "DetectionPredictor", "DuDetectionTrainer", "DetectionValidator"
