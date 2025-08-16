from TTS.api import TTS
import numpy as np
import soundfile as sf
from typing import List, Optional
import os

class VoiceCloner:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/your_tts"):
        """
        Initialize the voice cloner.
        
        Args:
            model_name: Name of the TTS model to use
        """
        self.tts = TTS(model_name)
        
    def extract_reference_audio(self, audio_data: np.ndarray, sample_rate: int,
                              segments: List[tuple], output_dir: str) -> List[str]:
        """
        Extract clean speech segments for voice cloning.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            segments: List of (start_time, end_time) tuples for clean speech
            output_dir: Directory to save reference audio files
            
        Returns:
            List of paths to saved reference audio files
        """
        os.makedirs(output_dir, exist_ok=True)
        reference_files = []
        
        for i, (start, end) in enumerate(segments):
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment = audio_data[start_sample:end_sample]
            
            output_path = os.path.join(output_dir, f"reference_{i}.wav")
            sf.write(output_path, segment, sample_rate)
            reference_files.append(output_path)
            
        return reference_files
        
    def synthesize_speech(self, text: str, reference_audio_paths: List[str],
                         output_path: Optional[str] = None, language: str = "en") -> np.ndarray:
        """
        Synthesize speech in the target voice.
        
        Args:
            text: Text to synthesize
            reference_audio_paths: List of paths to reference audio files
            output_path: Optional path to save the synthesized audio
            
        Returns:
            Synthesized audio data
        """
        # Use the first reference audio file for voice cloning
        reference_path = reference_audio_paths[0]
        
        print(f"!!!!!!!!!!!!Synthesizing speech in {language} language!!!!!!!!!!!!")

        # Synthesize speech
        if output_path:
            self.tts.tts_to_file(
                text=text,
                language=language,
                speaker_wav=reference_path,
                file_path=output_path
            )
            # Load the synthesized audio
            audio_data, sample_rate = sf.read(output_path)
        else:
            # Synthesize to memory
            audio_data = self.tts.tts(
                text=text,
                language=language,
                speaker_wav=reference_path
            )
            sample_rate = self.tts.synthesizer.output_sample_rate
            
        return audio_data, sample_rate 