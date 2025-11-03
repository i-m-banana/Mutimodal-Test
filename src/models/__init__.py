"""Model management wrappers and inference adapters."""

from .base_model import BaseModel
from .emotion_v2_model import EmotionV2Model
from .rgb_fatigue_model import RGBFatigueModel
from .eeg_fatigue_model import EEGFatigueModel
from .eeg_model import EEGModel
from .model_manager import ModelManager

__all__ = [
    "BaseModel",
    "EmotionV2Model",
    "RGBFatigueModel",
    "EEGFatigueModel",
    "EEGModel",
    "ModelManager",
]

# Note: STTModel is not yet implemented, stt_model.py is empty
