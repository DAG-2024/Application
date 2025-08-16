from transformers import pipeline
from typing import List, Dict, Tuple
import numpy as np

class TextPredictor:
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the text predictor.
        
        Args:
            model_name: Name of the language model to use
        """
        self.model = pipeline("fill-mask", model=model_name)
        
    def predict_text(self, context_before: str, context_after: str,
                    num_masks: int = 1) -> str:
        """
        Predict text to fill in a gap between two contexts.
        
        Args:
            context_before: Text before the gap
            context_after: Text after the gap
            num_masks: Number of [MASK] tokens to use
            
        Returns:
            Predicted text to fill the gap
        """
        # Create input with exactly one [MASK] token
        input_text = f"{context_before} [MASK] {context_after}"
        
        # Get predictions
        predictions = self.model(input_text)
        
        # Get the top prediction
        predicted_word = predictions[0]["token_str"]
        
        return predicted_word
    
    def predict_for_segment(self, word_segments: List[Dict],
                          segment_start: float, segment_end: float,
                          context_size: int = 5) -> str:
        """
        Predict text for a specific segment using surrounding context.
        
        Args:
            word_segments: List of word segments with timestamps
            segment_start: Start time of the segment to replace
            segment_end: End time of the segment to replace
            context_size: Number of words to use for context before and after
            
        Returns:
            Predicted text for the segment
        """
        # Find context words
        context_before = []
        context_after = []
        
        for segment in word_segments:
            if segment["end"] <= segment_start:
                context_before.append(segment["text"])
            elif segment["start"] >= segment_end:
                context_after.append(segment["text"])
                
        # Take last context_size words before and first context_size words after
        context_before = " ".join(context_before[-context_size:])
        context_after = " ".join(context_after[:context_size])
        
        # Predict text
        predicted_text = self.predict_text(context_before, context_after)
        
        return predicted_text 