import numpy as np
from typing import List, Dict, Tuple
import language_tool_python
from transformers import pipeline
import librosa
from sklearn.mixture import GaussianMixture

class ProblemSegmentDetector:
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the problem segment detector.
        
        Args:
            confidence_threshold: Threshold for ASR confidence scores
        """
        self.confidence_threshold = confidence_threshold
        self.language_tool = language_tool_python.LanguageTool('en-US')
        self.gmm = None
        
    def detect_low_confidence_segments(self, word_segments: List[Dict]) -> List[Tuple[float, float]]:
        """
        Detect segments with low ASR confidence.
        
        Args:
            word_segments: List of word segments with confidence scores
            
        Returns:
            List of (start_time, end_time) tuples for low confidence segments
        """
        problematic_segments = []
        current_segment = None
        
        for segment in word_segments:
            if segment["confidence"] < self.confidence_threshold:
                if current_segment is None:
                    current_segment = (segment["start"], segment["end"])
                else:
                    current_segment = (current_segment[0], segment["end"])
            else:
                if current_segment is not None:
                    problematic_segments.append(current_segment)
                    current_segment = None
                    
        if current_segment is not None:
            problematic_segments.append(current_segment)
            
        return problematic_segments
    
    def detect_grammar_errors(self, text: str, word_segments: List[Dict]) -> List[Tuple[float, float]]:
        """
        Detect segments with grammatical errors.
        
        Args:
            text: Full transcript text
            word_segments: List of word segments with timestamps
            
        Returns:
            List of (start_time, end_time) tuples for segments with grammar errors
        """
        matches = self.language_tool.check(text)
        problematic_segments = []
        
        for match in matches:
            # Find the word segments that overlap with the error
            error_start = match.offset
            error_end = match.offset + match.errorLength
            
            # Convert character positions to time
            start_time = None
            end_time = None
            current_pos = 0
            
            for segment in word_segments:
                segment_length = len(segment["text"])
                if current_pos <= error_start < current_pos + segment_length:
                    start_time = segment["start"]
                if current_pos <= error_end < current_pos + segment_length:
                    end_time = segment["end"]
                    break
                current_pos += segment_length + 1  # +1 for space
                
            if start_time is not None and end_time is not None:
                problematic_segments.append((start_time, end_time))
                
        return problematic_segments
    
    def train_acoustic_model(self, audio_data: np.ndarray, sample_rate: int,
                           clean_segments: List[Tuple[float, float]]):
        """
        Train a GMM on clean speech segments for acoustic anomaly detection.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            clean_segments: List of (start_time, end_time) tuples for clean speech
        """
        if len(clean_segments) < 2:
            print("Warning: Not enough clean segments for GMM training. Using default model.")
            self.gmm = None
            return
            
        # Extract MFCC features from clean segments
        mfcc_features = []
        
        for start, end in clean_segments:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment = audio_data[start_sample:end_sample]
            
            mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate)
            mfcc_features.append(mfcc.T)
            
        # Concatenate all features
        all_features = np.vstack(mfcc_features)
        
        # Train GMM
        self.gmm = GaussianMixture(n_components=min(16, len(clean_segments)))
        self.gmm.fit(all_features)
        
    def detect_acoustic_anomalies(self, audio_data: np.ndarray, sample_rate: int,
                                window_size: float = 0.1) -> List[Tuple[float, float]]:
        """
        Detect segments with acoustic anomalies using the trained GMM.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            window_size: Size of analysis window in seconds
            
        Returns:
            List of (start_time, end_time) tuples for anomalous segments
        """
        if self.gmm is None:
            print("Warning: GMM not trained. Skipping acoustic anomaly detection.")
            return []
            
        window_samples = int(window_size * sample_rate)
        problematic_segments = []
        current_segment = None
        
        for i in range(0, len(audio_data) - window_samples, window_samples):
            window = audio_data[i:i + window_samples]
            mfcc = librosa.feature.mfcc(y=window, sr=sample_rate).T
            
            # Calculate log-likelihood
            score = self.gmm.score_samples(mfcc)
            avg_score = np.mean(score)
            
            # If score is significantly lower than average, mark as problematic
            if avg_score < -10:  # Threshold needs tuning
                time = i / sample_rate
                if current_segment is None:
                    current_segment = (time, time + window_size)
                else:
                    current_segment = (current_segment[0], time + window_size)
            else:
                if current_segment is not None:
                    problematic_segments.append(current_segment)
                    current_segment = None
                    
        if current_segment is not None:
            problematic_segments.append(current_segment)
            
        return problematic_segments