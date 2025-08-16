from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
from typing import Dict, List, Tuple
import numpy as np
import librosa
import os

class WhisperASR:
    def __init__(self, model_name: str = "openai/whisper-large-v3", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_cuda: bool = True):
        """
        Initialize Whisper ASR model.
        
        Args:
            model_name: Name of the Whisper model to use
            device: Device to run the model on ("cuda" or "cpu")
            use_cuda: Whether to use CUDA if available (overrides device if True and CUDA is available)
        """
        if use_cuda and torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA for ASR")
        else:
            device = "cpu"
            print("Using CPU for ASR")
            
        # Initialize processor and model directly for more control
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device
        self.target_sr = 16000  # Whisper expects 16kHz audio
        
    def transcribe(self, audio_data: np.ndarray, sample_rate: int,
                  language: str = "english") -> Dict:
        """
        Transcribe audio data using Whisper.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate of the audio
            language: Language of the audio
            
        Returns:
            Dictionary containing transcription results
        """
        # Convert audio data to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Resample to 16kHz if needed
        if sample_rate != self.target_sr:
            print(f"Resampling audio from {sample_rate}Hz to {self.target_sr}Hz...")
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=self.target_sr
            )
            sample_rate = self.target_sr
        
        # Process audio
        input_features = self.processor(
            audio_data, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate transcription with timestamps
        predicted_ids = self.model.generate(
            input_features,
            language=language,
            task="transcribe",
            return_timestamps=True,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        with open(os.path.join("output/steps", "3.1_transcription.txt"), "w") as f:
            f.write(str(predicted_ids))
        
        # Get the generated sequence and ensure it's the right shape
        if isinstance(predicted_ids, dict):
            sequence = predicted_ids["sequences"]
        else:
            sequence = predicted_ids
            
        # Ensure sequence is 2D (batch_size, sequence_length)
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)
        
        # Decode the transcription
        transcription = self.processor.batch_decode(
            sequence, 
            skip_special_tokens=False
        )[0]
        
        # Get timestamps
        timestamps = self.processor.batch_decode(
            sequence, 
            skip_special_tokens=False,
            decode_with_timestamps=True
        )[0]
        
        # Process the output to get word-level timestamps
        chunks = []
        current_text = ""
        current_start = None
        current_end = None
        
        for token in timestamps.split():
            if token.startswith("<|") and token.endswith("|>"):
                if current_start is not None and current_text:
                    chunks.append({
                        "text": current_text.strip(),
                        "timestamp": (current_start, current_end)
                    })
                    current_text = ""
                if token == "<|notimestamps|>":
                    continue
                try:
                    time = float(token.strip("<|>"))
                    if current_start is None:
                        current_start = time
                    current_end = time
                except ValueError:
                    continue
            else:
                current_text += " " + token
        
        # Add the last chunk if exists
        if current_text and current_start is not None:
            chunks.append({
                "text": current_text.strip(),
                "timestamp": (current_start, current_end)
            })
        
        return {
            "text": transcription,
            "chunks": chunks
        }
    
    def get_word_segments(self, transcription_result: Dict) -> List[Dict]:
        """
        Extract word segments from transcription result.
        
        Args:
            transcription_result: Result from transcribe method
            
        Returns:
            List of word segments with text, start time, end time, and confidence
        """
        word_segments = []
        
        if "chunks" in transcription_result:
            for chunk in transcription_result["chunks"]:
                if "timestamp" in chunk:
                    start, end = chunk["timestamp"]
                    word_segments.append({
                        "text": chunk["text"].strip(),
                        "start": start,
                        "end": end,
                        "confidence": 1.0  # Whisper doesn't provide confidence scores
                    })
                    
        return word_segments 