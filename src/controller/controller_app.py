import sys
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from typing import List
import re
import uvicorn


log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

app_logger = logging.getLogger("controller_app")
app_logger.setLevel(logging.DEBUG)

if not app_logger.handlers:
    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    app_logger.addHandler(file_handler)
    app_logger.addHandler(stream_handler)

# Ensure the parent directory is in the path to import controllerUtils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controllerUtils import (
    transcribe,
    detect_energy,
    get_words_in_loud_segments,
    word_overlap_with_noise,
    ctx_anomaly_detector,
    word_predictor,
    build_word_tokens_of_detection,
    plot_speech_spectrogram,
)

from src.stitcher.models.stitcherModels import WordToken, wordtokens_to_json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to save uploaded audio files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "input", "uploaded_audio")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/feed-audio", response_model=dict[str, list[WordToken]])
async def feed_audio(file: UploadFile = File(...)):
    """
    Endpoint to upload and save an original audio file. Also runs the previous main logic for future enhancement.
    Now returns a list of WordToken objects as JSON, with to_synth=True for predicted [blank] words.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    AUDIO_PATH = file_path
    
    try:
        whisper_result = transcribe(AUDIO_PATH)
        whisper_transcription = segments_to_transcription(whisper_result)
        indexed_transcription = indexed_transcription_str(whisper_transcription)

        app_logger.debug(f"Whisper:\t{whisper_transcription}")

        anomaly_res = ctx_anomaly_detector(whisper_transcription, indexed_transcription)
        anomaly_idxs_str = (anomaly_res.choices[0].message.content or "").strip()
        anomaly_idxs = parse_indices_string(anomaly_idxs_str) if anomaly_idxs_str else []

        tokens = build_word_tokens_of_detection(
            wav_path=AUDIO_PATH,
            anomaly_word_idx=anomaly_idxs,
            whisper_json_or_path=whisper_result,
            low_conf_th = 0.3,         # confidence threshold for low-confidence words

            w_conf_w = 0.59,            # weight for confidence term (low confidence => more likely to synth)
            energy_w = 0.67,            # weight for energy overlap term
            anomaly_w = 0.66,           # weight for anomaly term
            synth_score_th = 0.60,      # threshold to decide synthesis

            gap_min_dur = 0.12,         # ignore tiny gaps
            gap_energy_cov_th = 0.30,   # fraction of gap covered by energy to consider it noise
            gap_energy_score_th = 0.75,  # min energy score to consider gap as noise

            plot_spectrogram = True     # plot spectrogram for logging/debugging
        )

        template = " ".join([t.text for t in tokens])
        app_logger.debug(f"Template:\t{template}")
        predicted = word_predictor(template)
        app_logger.debug(f"Predicted:\t{predicted}")
        wordtokens = align_blanks_and_predicted(tokens, predicted)


        return {"wordtokens": wordtokens}
    
    except Exception as e:
        app_logger.error(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Controller is running",
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint to receive an audio file and return its transcription.
    """
    temp_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = transcribe(temp_path)
        transcription = segments_to_transcription(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return JSONResponse(content={"transcription": transcription})

# To run: uvicorn src.controller.main:app --reload

#------------Helper Functions------------

def segments_to_transcription(data):
    """
    Given WhisperX-style segmented output with word-level timing and scores,
    return the full raw transcription as a plain string.
    """
    words = []
    for segment in data.get('segments', []):
        for word_info in segment.get('words', []):
            words.append(word_info['word'])
    return ' '.join(words)

def indexed_transcription_str(text):

    """
    Returns a string where each word in the input text is followed by its index.
    Format: '0: word\n1: word\n...'
    """
    words = text.strip().split()
    lines = [f"{idx}: {word}" for idx, word in enumerate(words)]
    return "\n".join(lines)

def parse_indices_string(s):
    return [int(num.strip()) for num in s.split(',')]

def align_blanks_and_predicted(pre_tokens: List[WordToken], predicted_text):
    pred_words = predicted_text.strip().split()

    wordtokens = []
    orig_idx = 0
    pred_idx = 0

    while orig_idx < len(pre_tokens) and pred_idx < len(pred_words):
        w = pre_tokens[orig_idx]

        if w.text == '[blank]':
            # Handle blank tokens - find next non-blank to determine boundary
            next_non_blank_idx = None
            for j in range(orig_idx + 1, len(pre_tokens)):
                if pre_tokens[j].text != '[blank]':
                    next_non_blank_idx = j
                    break

            # Collect predicted words until we hit the next non-blank match
            blank_phrase = []
            while pred_idx < len(pred_words):
                if next_non_blank_idx is not None:
                    # Check if current predicted word matches the next non-blank
                    if _words_match(pred_words[pred_idx], pre_tokens[next_non_blank_idx].text):
                        break
                    # Check if combined predicted words match the next non-blank
                    combined = ' '.join(blank_phrase + [pred_words[pred_idx]])
                    if _words_match(combined.replace(' ', ''), pre_tokens[next_non_blank_idx].text):
                        blank_phrase.append(pred_words[pred_idx])
                        pred_idx += 1
                        break

                blank_phrase.append(pred_words[pred_idx])
                pred_idx += 1

            # Add synthesized words for the blank
            for word in blank_phrase:
                wordtokens.append(WordToken(
                    start=w.start,
                    end=w.end,
                    text=word,
                    to_synth=True,
                    is_speech=True,
                    synth_path=None
                ))
            orig_idx += 1

        else:
            # Handle non-blank tokens - check for exact match, combination, or change
            if _words_match(pred_words[pred_idx], w.text):
                # Exact match
                wordtokens.append(WordToken(
                    start=w.start,
                    end=w.end,
                    text=pred_words[pred_idx],
                    to_synth=False,
                    is_speech=True,
                    synth_path=None
                ))
                pred_idx += 1
                orig_idx += 1

            else:
                # Check if multiple predicted words combine to match original
                combined_pred = pred_words[pred_idx]
                temp_pred_idx = pred_idx + 1

                while temp_pred_idx < len(pred_words) and not _words_match(combined_pred, w.text):
                    combined_pred += pred_words[temp_pred_idx]
                    temp_pred_idx += 1

                if _words_match(combined_pred, w.text):
                    # Multiple predicted words match one original - use original timing
                    wordtokens.append(WordToken(
                        start=w.start,
                        end=w.end,
                        text=combined_pred,
                        to_synth=False,
                        is_speech=True,
                        synth_path=None
                    ))
                    pred_idx = temp_pred_idx
                    orig_idx += 1

                else:
                    # Check if one predicted word matches multiple original words
                    combined_orig = format_text(w.text)
                    temp_orig_idx = orig_idx + 1

                    while temp_orig_idx < len(pre_tokens) and pre_tokens[temp_orig_idx].text != '[blank]':
                        combined_orig += format_text(pre_tokens[temp_orig_idx].text)
                        if _words_match(pred_words[pred_idx], combined_orig):
                            # One predicted word matches multiple original words
                            wordtokens.append(WordToken(
                                start=w.start,
                                end=pre_tokens[temp_orig_idx].end,
                                text=pred_words[pred_idx],
                                to_synth=True,  # Changed word, needs synthesis
                                is_speech=True,
                                synth_path=None
                            ))
                            pred_idx += 1
                            orig_idx = temp_orig_idx + 1
                            break
                        temp_orig_idx += 1
                    else:
                        # No match found - word was changed, needs synthesis
                        wordtokens.append(WordToken(
                            start=w.start,
                            end=w.end,
                            text=pred_words[pred_idx],
                            to_synth=True,
                            is_speech=True,
                            synth_path=None
                        ))
                        pred_idx += 1
                        orig_idx += 1

    # Handle remaining predicted words if any
    while pred_idx < len(pred_words):
        # Use timing from last token if available
        last_end = wordtokens[-1].end if wordtokens else 0.0
        wordtokens.append(WordToken(
            start=last_end,
            end=last_end,
            text=pred_words[pred_idx],
            to_synth=True,
            is_speech=True,
            synth_path=None
        ))
        pred_idx += 1

    return wordtokens


def _words_match(word1, word2):
    """
    Check if two words match, handling common variations like:
    - Case differences
    - Punctuation
    - Space combinations (e.g., "with in" vs "within")
    """
    # Normalize both words
    norm1 = re.sub(r'[^\w]', '', word1.lower())
    norm2 = re.sub(r'[^\w]', '', word2.lower())

    # Direct match
    if norm1 == norm2:
        return True

    # Check if one is a space-separated version of the other
    spaced1 = re.sub(r'([a-z])([A-Z])', r'\1 \2', word1.lower()).replace(' ', '')
    spaced2 = re.sub(r'([a-z])([A-Z])', r'\1 \2', word2.lower()).replace(' ', '')

    return spaced1 == spaced2

def format_text(text):
    t = re.sub(r'[\.!\?,\'\"\*;:]+$', '', text).lower()
    return t