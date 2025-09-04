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

        anomaly_res = ctx_anomaly_detector(whisper_transcription, indexed_transcription)
        anomaly_idxs_str = (anomaly_res.choices[0].message.content or "").strip()
        anomaly_idxs = parse_indices_string(anomaly_idxs_str) if anomaly_idxs_str else []

        tokens = build_word_tokens_of_detection(
            wav_path=AUDIO_PATH,
            anomaly_word_idx=anomaly_idxs,
            whisper_json_or_path=whisper_result,
            low_conf_th = 0.3,         # confidence threshold for low-confidence words

            w_conf_w = 0.50,            # weight for confidence term (low confidence => more likely to synth)
            energy_w = 0.40,            # weight for energy overlap term
            anomaly_w = 0.60,           # weight for anomaly term
            synth_score_th = 0.60,      # threshold to decide synthesis

            gap_min_dur = 0.12,         # ignore tiny gaps
            gap_energy_cov_th = 0.30,   # fraction of gap covered by energy to consider it noise

            plot_spectrogram = True     # plot spectrogram for logging/debugging
        )

        template = " ".join([t.text for t in tokens])
        predicted = word_predictor(template)
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

def print_low_confidence_words(whisper_res, thresh):
    print(f"Words below {thresh} confidence score:\n")
    for segment in whisper_res.get("segments", []):
        for word_info in segment.get("words", []):
            if word_info.get("score", 1.0) < thresh:
                print(f"'{word_info['word']}'  --  score: '{word_info['score']:.2f}'")

def print_loud_noise_segments(segments):
    print("Loud noise detection: \n")
    for segment in segments:
        if segment.get('label') == 'loud_noise':
            print(segment)

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

def intersect_with_low_confidence_score(whisper_res, wrong_words_indices: list, thresh=0.3):
    """
    Given a WhisperX-style transcript and a list of word indexes,
    return a list of indexes where the word's confidence score < 0.3.
    """
    all_words = []
    for segment in whisper_res.get('segments', []):
        all_words.extend(segment.get('words', []))

    return [
        idx for idx in wrong_words_indices
        if 0 <= idx < len(all_words) and all_words[idx].get('score', 1.0) < thresh
    ]

def adjust_by_noise_segments(all_words, loud_noise_segments):
    """
    Given a flat list of words (all_words) and a list of loud noise intervals,
    replaces words that overlap with '[blank]'. If no words overlap a noise interval,
    inserts a new '[blank]' word in the appropriate position in all_words.

    Modifies all_words in-place.
    """
    # Sort all_words by start time to enable ordered insertion
    all_words.sort(key=lambda w: float(w.start))

    for noise in loud_noise_segments:
        if noise['label'] == 'loud_noise':
            start = float(noise['start_time'])
            end = float(noise['end_time'])

            matched = False
            for word in all_words:
                word_start = float(word['start'])
                word_end = float(word['end'])

                if word_start < end and word_end > start:
                    word['word'] = '[blank]'
                    word['start'] = max(word_start, start)
                    word['end'] = min(word_end, end)
                    matched = True

            if not matched:
                # Insert a new '[blank]' word into the correct position
                new_word = {'word': '[blank]', 'start': start, 'end': end, 'score': 0.0}
                inserted = False
                for i, word in enumerate(all_words):
                    if float(word['start']) > end:
                        all_words.insert(i, new_word)
                        inserted = True
                        break
                if not inserted:
                    all_words.append(new_word)

    # Ensure all_words remains sorted
    all_words.sort(key=lambda w: float(w.start))
    return all_words


def insert_blank_and_modify_timestamp(whisper_res, indexes):
    """
    Adjust timestamps of words at given indexes:
    - start time is set to end time of previous word
    - end time is set to start time of next word

    Modifies the transcript in place.
    """
    # Flatten words into a list of (word_dict, segment_ref)
    all_words = []
    for segment in whisper_res.get('segments', []):
        for word in segment.get('words', []):
            all_words.append(word)

    for idx in indexes:
        if 1 <= idx < len(all_words) - 1:
            prev_word = all_words[idx - 1]
            curr_word = all_words[idx]
            next_word = all_words[idx + 1]

            curr_word['start'] = float(prev_word['end'])
            curr_word['word'] = '[blank]'
            curr_word['end'] = float(next_word['start'])
        # Optional: skip edge cases (first or last word) silently
    return all_words

def align_blanks_and_predicted(pre_tokens: List[WordToken], predicted_text):
    pred_words = predicted_text.strip().split()

    wordtokens = []
    orig_idx = 0
    pred_idx = 0
    while orig_idx < len(pre_tokens):
        w = pre_tokens[orig_idx]
        if w.text == '[blank]':
            # Find the next non-blank word in the original
            next_non_blank = None
            for j in range(orig_idx + 1, len(pre_tokens)):
                if pre_tokens[j].text != '[blank]':
                    next_non_blank = pre_tokens[j].text
                    break
            phrase = []
            while pred_idx < len(pred_words) and (next_non_blank is None or pred_words[pred_idx] != next_non_blank):
                phrase.append(pred_words[pred_idx])
                pred_idx += 1
            for word in phrase:
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
            # If the predicted word matches, consume it
            if pred_idx < len(pred_words) and re.sub(r'[\.!\?,;:]+$', '', pred_words[pred_idx]) == re.sub(r'[\.!\?,;:]+$', '', w.text):
                wordtokens.append(WordToken(
                    start=w.start,
                    end=w.end,
                    text=pred_words[pred_idx],
                    to_synth=False,
                    is_speech=True,
                    synth_path=None
                ))
                pred_idx += 1
            else:
                # If not matching, just add the original word
                wordtokens.append(WordToken(
                    start=w.start,
                    end=w.end,
                    text=w.text,
                    to_synth=False,
                    is_speech=True,
                    synth_path=None
                ))
            orig_idx += 1

    return wordtokens
