# TRANSCRIBE AUDIO AND IDENTIFY GAPS BETWEEN WORDS
# This script transcribes an audio file using Faster Whisper and provides word-level timing.
# TODO:
# 1. Adjust script to get audio files by HTTP request and return the result as a JSON response.
# 2. Consider adjusting model parameters for better performance. e.g., device, quantization, model size, compute type, etc.
# 3. Work on evaluating the performance of the script. e.g., accuracy, speed, etc.
# 4. Add error handling for file loading and processing.
# 5. Add logging for better debugging and monitoring.
# 6. Consider testing the handling of different audio formats.
# 7. Consider testing the handling of other languages.

import whisperx
import torch
import os

def transcribe(audio_path):
    # === 1. Transcribe with WhisperX ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("tiny.en", device, compute_type="float32")

    result = model.transcribe(audio_path)
    
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)

    return result_aligned