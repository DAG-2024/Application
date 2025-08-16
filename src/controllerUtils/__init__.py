"""
ControllerUtils module for audio processing and analysis.

This module provides utilities for:
- Audio transcription using Whisper
- Energy-based noise detection
- Noisy word detection and analysis
- Context anomaly detection
- Word prediction and completion
"""

from .transcriber import transcribe
from .energy_scorer import detect_energy
from .noisy_words_detector import get_words_in_loud_segments, word_overlap_with_noise
from .context_anomaly_detection import ctx_anomaly_detector
from .predictor import word_predictor

__all__ = [
    'transcribe',
    'detect_energy', 
    'get_words_in_loud_segments',
    'word_overlap_with_noise',
    'ctx_anomaly_detector',
    'word_predictor'
]
