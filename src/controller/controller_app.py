from fastapi import Form
import requests
from stitcher.models.stitcherModels import wordtokens_from_json
import sys
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [
        logging.FileHandler("app.log"),  # Logs will be written to 'app.log'
        logging.StreamHandler()  # Logs will still be printed to the console
    ]
)
logger = logging.getLogger(__name__)

# Ensure the parent directory is in the path to import controllerUtils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllerUtils import (
    transcribe,
    detect_energy,
    get_words_in_loud_segments,
    word_overlap_with_noise,
    ctx_anomaly_detector,
    word_predictor
)
from stitcher.models.stitcherModels import WordToken, wordtokens_to_json

app = FastAPI()

# Directory to save uploaded audio files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "input", "uploaded_audio")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Endpoint to receive edited transcription and return fixed audio

@app.post("/fix-edited-audio")
async def fix_edited_audio(
    file: UploadFile = File(...),
    original_wordtokens: str = Form(...),
    original_transcription: str = Form(...),
    edited_transcription: str = Form(...)
):
    """
    Receives the edited transcription and original wordtokens list from the UI, compares to original transcription,
    reassembles the wordtokens list (changed/new words marked for synthesis), sends to stitcher, returns fixed audio URL.
    """
    try:
        # Save uploaded audio
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse original wordtokens
        orig_wordtokens = wordtokens_from_json(original_wordtokens)

        # Reassemble wordtokens list using a robust alignment algorithm
        new_wordtokens = align_transcriptions(
            orig_wordtokens,
            original_transcription,
            edited_transcription
        )

        # Serialize new wordtokens for stitcher
        new_wordtokens_json = wordtokens_to_json(new_wordtokens)

        # Send to stitcher fix-audio endpoint
        stitcher_url = "http://localhost:8000/fix-audio" # CHANGE PORT (according to config file)
        with open(file_path, "rb") as audio_file:
            response = requests.post(
                stitcher_url,
                files={"file": (file.filename, audio_file, file.content_type)},
                data={"payload": new_wordtokens_json}
            )
        response.raise_for_status()
        fixed_url = response.json().get("fixed_url")
        return JSONResponse(content={"fixed_url": fixed_url})
    except Exception as e:
        logger.error(f"Error in fix-edited-audio: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/feed-audio")
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
    CONF_THRESH = 0.3
    USE_LOW_CONF_INTERSECTION = True
    USE_NOISE_MASKING = True
    
    try:
        whisper_result = transcribe(AUDIO_PATH)
        whisper_transcription = segments_to_transcription(whisper_result)
        indexed_transcription = indexed_transcription_str(whisper_transcription)


        anomaly_res = ctx_anomaly_detector(whisper_transcription, indexed_transcription)
        anomaly_idxs_str = (anomaly_res.choices[0].message.content or "").strip()

        anomaly_idxs = parse_indices_string(anomaly_idxs_str) if anomaly_idxs_str else []
        if USE_LOW_CONF_INTERSECTION:
            anomaly_idxs_intersect = intersect_with_low_confidence_score(whisper_result, anomaly_idxs, thresh=CONF_THRESH)
        else:
            anomaly_idxs_intersect = anomaly_idxs


        words_after_anomaly_mask = insert_blank_and_modify_timestamp(whisper_result, anomaly_idxs_intersect)

        if USE_NOISE_MASKING:
            energy_segments = detect_energy(AUDIO_PATH)
            words_after_noise_mask = adjust_by_noise_segments(words_after_anomaly_mask, energy_segments)
        else:
            energy_segments = []
            words_after_noise_mask = words_after_anomaly_mask


        # Pre-prediction WordToken list (with blanks)
        pre_wordtokens = []
        for w in words_after_noise_mask:
            pre_wordtokens.append(WordToken(
                start=w['start'],
                end=w['end'],
                text=w['word'],
                to_synth=(w['word'] == '[blank]'),
                is_speech=True,
                synth_path=None
            ))

        blank_inserted_trans = ' '.join(w['word'] for w in words_after_noise_mask)
        predicted_text = word_predictor(blank_inserted_trans)

        # Post-prediction WordToken list (with predicted words)
        post_wordtokens = align_blanks_and_predicted(words_after_noise_mask, predicted_text)

        return JSONResponse(content={
            "pre_wordtokens": wordtokens_to_json(pre_wordtokens),
            "post_wordtokens": wordtokens_to_json(post_wordtokens),
            "pre_transcription": blank_inserted_trans,
            "post_transcription": predicted_text
        })
    except Exception as e:
        logger.error(f"Error: {str(e)}")
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
    all_words.sort(key=lambda w: float(w['start']))

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
    all_words.sort(key=lambda w: float(w['start']))
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

def align_blanks_and_predicted(words_after_noise_mask, predicted_text):
    """
    Simplified alignment assuming:
    - Each [blank] is replaced by 1-2 words only
    - No consecutive [blank]s
    - Timing doesn't matter for to_synth=True words
    """
    orig_words = [w['word'] for w in words_after_noise_mask]
    pred_words = predicted_text.strip().split()
    
    # Normalize words for comparison (case-insensitive only, preserve punctuation)
    def normalize_word(word):
        return word.lower()
    
    wordtokens = []
    orig_idx = 0
    pred_idx = 0
    
    while orig_idx < len(words_after_noise_mask) and pred_idx < len(pred_words):
        w = words_after_noise_mask[orig_idx]
        
        if w['word'] == '[blank]':
            # Since each blank is 1-2 words, we need to determine how many
            # Look ahead to find the next non-blank word as an anchor
            next_anchor = None
            next_anchor_idx = orig_idx + 1
            
            while next_anchor_idx < len(orig_words):
                if orig_words[next_anchor_idx] != '[blank]':
                    next_anchor = normalize_word(orig_words[next_anchor_idx])
                    break
                next_anchor_idx += 1
            
            # Consume 1-2 words until we hit the anchor or reach max
            consumed_words = []
            max_words_for_blank = 2
            
            while (pred_idx < len(pred_words) and 
                   len(consumed_words) < max_words_for_blank):
                
                current_pred_word = normalize_word(pred_words[pred_idx])
                
                # If we found our anchor, stop consuming
                if next_anchor and current_pred_word == next_anchor:
                    break
                
                consumed_words.append(pred_words[pred_idx])
                pred_idx += 1
            
            # Create WordTokens for consumed words (timing doesn't matter)
            for word in consumed_words:
                wordtokens.append(WordToken(
                    start=w['start'],  # Same timing is fine
                    end=w['end'],
                    text=word,
                    to_synth=True,
                    is_speech=True,
                    synth_path=None
                ))
            
        else:
            # Non-blank word - should match exactly (case-insensitive only)
            if pred_idx < len(pred_words):
                pred_normalized = normalize_word(pred_words[pred_idx])
                orig_normalized = normalize_word(w['word'])
                
                if pred_normalized == orig_normalized:
                    # Use the predicted word (may have different capitalization)
                    wordtokens.append(WordToken(
                        start=w['start'],
                        end=w['end'],
                        text=pred_words[pred_idx],
                        to_synth=False,
                        is_speech=True,
                        synth_path=None
                    ))
                    pred_idx += 1
                else:
                    # Mismatch - use original word and log warning
                    print(f"Warning: Word mismatch at position {orig_idx}. "
                          f"Expected '{w['word']}', got '{pred_words[pred_idx] if pred_idx < len(pred_words) else 'END'}'")
                    wordtokens.append(WordToken(
                        start=w['start'],
                        end=w['end'],
                        text=w['word'],
                        to_synth=False,  # Keep original
                        is_speech=True,
                        synth_path=None
                    ))
            else:
                # Ran out of predicted words - use original
                wordtokens.append(WordToken(
                    start=w['start'],
                    end=w['end'],
                    text=w['word'],
                    to_synth=False,
                    is_speech=True,
                    synth_path=None
                ))
        
        orig_idx += 1
    
    # Handle remaining original words (if predicted text was shorter)
    while orig_idx < len(words_after_noise_mask):
        w = words_after_noise_mask[orig_idx]
        wordtokens.append(WordToken(
            start=w['start'],
            end=w['end'],
            text=w['word'] if w['word'] != '[blank]' else '[MISSING]',
            to_synth=(w['word'] == '[blank]'),  # Mark blanks for synthesis
            is_speech=True,
            synth_path=None
        ))
        orig_idx += 1
    
    # Log warning if there are leftover predicted words
    if pred_idx < len(pred_words):
        leftover = pred_words[pred_idx:]
        print(f"Warning: {len(leftover)} unused predicted words: {leftover}")
    
    return wordtokens

def align_transcriptions(orig_tokens, orig_transcription, edited_transcription):
    """
    Aligns the original and edited transcriptions using a simple dynamic programming
    approach to handle insertions, deletions, and substitutions.
    Returns a new list of WordToken objects marked for synthesis.
    """
    orig_words = orig_transcription.strip().split()
    edited_words = edited_transcription.strip().split()

    # Create a new list for the output WordTokens
    new_wordtokens = []
    
    # Use two pointers for alignment
    orig_idx = 0
    edited_idx = 0
    
    while edited_idx < len(edited_words) or orig_idx < len(orig_words):
        edited_word = edited_words[edited_idx] if edited_idx < len(edited_words) else None
        orig_word = orig_words[orig_idx] if orig_idx < len(orig_words) else None
        orig_token = orig_tokens[orig_idx] if orig_idx < len(orig_tokens) else None

        # Case 1: Word matches, use original WordToken
        if edited_word and orig_word and edited_word.lower() == orig_word.lower():
            new_wordtokens.append(WordToken(
                start=orig_token.start,
                end=orig_token.end,
                text=edited_word,
                to_synth=False,
                is_speech=True,
                synth_path=None
            ))
            orig_idx += 1
            edited_idx += 1
            
        # Case 2: Word was inserted in edited transcription
        elif edited_word and (not orig_word or edited_idx > orig_idx):
            # Assign a default duration and mark for synthesis
            last_end = new_wordtokens[-1].end if new_wordtokens else 0.0
            new_token = WordToken(
                start=last_end,
                end=last_end + 0.5, # Default duration
                text=edited_word,
                to_synth=True,
                is_speech=True,
                synth_path=None
            )
            new_wordtokens.append(new_token)
            edited_idx += 1
            
        # Case 3: Word was deleted from original transcription
        elif orig_word and (not edited_word or orig_idx > edited_idx):
            # Skip this word and advance the original pointer
            orig_idx += 1

        # Case 4: Mismatch (substitution or more complex change)
        else:
            # Mark the edited word for synthesis
            new_token = WordToken(
                start=orig_token.start,
                end=orig_token.end,
                text=edited_word,
                to_synth=True,
                is_speech=True,
                synth_path=None
            )
            new_wordtokens.append(new_token)
            orig_idx += 1
            edited_idx += 1

    return new_wordtokens
