# Audio Processing and Transcription Project

This project provides audio transcription, noise detection, and word prediction capabilities using various AI models and audio processing libraries.

## Features

- **Audio Transcription**: Uses Faster Whisper for high-quality speech-to-text conversion with word-level timestamps
- **Energy-based Noise Detection**: Identifies loud noise segments in audio using librosa
- **Context Anomaly Detection**: Uses OpenAI GPT-4 to detect contextually incorrect words
- **Word Prediction**: Predicts missing words in transcriptions using AI
- **Noisy Word Detection**: Identifies words that overlap with noise segments

## Project Structure

```
src/
├── controller/
│   └── main.py                 # Main application entry point
├── controllerUtils/            # Core processing modules
│   ├── __init__.py            # Module initialization
│   ├── transcriber.py         # Audio transcription using Faster Whisper
│   ├── energy_scorer.py       # Energy-based noise detection
│   ├── noisy_words_detector.py # Word-noise overlap detection
│   ├── context_anomaly_detection.py # Context anomaly detection
│   └── predictor.py           # Word prediction
├── input/                     # Input audio files
├── output/                    # Output files
└── unused/                    # Unused/experimental modules
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Navigate to the project directory
cd /path/to/your/project/src

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Run the test script to verify everything is working
python test_setup.py
```

### 4. Environment Variables

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Main Application

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Run the main application
python controller/main.py
```

### Using Individual Modules

```python
from controllerUtils import (
    transcribe,
    detect_energy,
    ctx_anomaly_detector,
    word_predictor
)

# Transcribe audio
result = transcribe("path/to/audio.wav")

# Detect energy segments
energy_segments = detect_energy("path/to/audio.wav")

# Detect context anomalies
anomalies = ctx_anomaly_detector(transcription, indexed_transcription)

# Predict missing words
predicted_text = word_predictor(transcription_with_blanks)
```

## Dependencies

### Core Dependencies
- **faster-whisper**: Fast speech recognition using Whisper models
- **torch**: PyTorch for deep learning operations
- **librosa**: Audio processing and analysis
- **numpy**: Numerical computing
- **openai**: OpenAI API client for GPT models
- **python-dotenv**: Environment variable management

### Optional Dependencies (for unused modules)
- speechbrain
- torchaudio
- voicefixer
- language-tool-python
- transformers
- scikit-learn
- TTS
- soundfile
- datasets

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure your virtual environment is activated
2. **CUDA Issues**: The code automatically falls back to CPU if CUDA is not available
3. **API Key Issues**: Ensure your OpenAI API key is set in the `.env` file
4. **Audio Format Issues**: The system supports common audio formats (WAV, MP3, etc.)

### Performance Tips

- Use GPU acceleration if available (CUDA)
- Adjust model sizes in `transcriber.py` for speed vs. accuracy trade-offs
- Consider using smaller Whisper models for faster processing

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!
