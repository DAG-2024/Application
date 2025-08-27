import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Form
from .models.stitcherModels import WordToken, FixResponse
from typing import List
import uuid, os, requests
import ffmpeg
import logging
import re
from tempfile import gettempdir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [
        logging.FileHandler("stitcher.log"),  # Logs will be written to 'app.log'
        logging.StreamHandler()  # Logs will still be printed to the console
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()


async def save_upload_file(upload: UploadFile) -> str:
    """Save uploaded UploadFile to a tmp file and return its path."""
    ext = os.path.splitext(upload.filename)[1] or ".wav"
    path = os.path.join(gettempdir(), uuid.uuid4().hex + ext)
    data = await upload.read()
    with open(path, "wb") as f:
        f.write(data)
    return path

def normalize_text(text: str, default_punct: str = '.') -> str:
    text = text.strip()
    # Check if ends with ? or !
    if text.endswith('?') or text.endswith('!'):
        punct = text[-1]
        text = text[:-1]
    else:
        punct = default_punct
        text = re.sub(r'[\.!\?,;:]+$', '', text)
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    return text + punct

def mean_word_gap(wordtokens: list) -> float:
    """
    Calculate the mean gap (in seconds) between consecutive WordToken objects.
    Returns 0.0 if fewer than 2 tokens.
    """
    if len(wordtokens) < 2:
        return 0.0
    gaps = [
        wordtokens[i].start - wordtokens[i - 1].end
        for i in range(1, len(wordtokens))
        if wordtokens[i].start > wordtokens[i - 1].end
    ]
    return sum(gaps) / len(gaps) if gaps else 0.0

def make_voice_print(orig_path: str, wordTokens: List[WordToken]) -> (str, str):
    """
    Extract the longest sequential clean speech segment (up to 15s) from the original audio using WordToken timings.
    """
    # Find all runs of clean speech
    runs = []
    current_run = []
    best_run = None

    # Max gap between two words for voice print
    gap = mean_word_gap(wordTokens) * 2

    for s in wordTokens:
        if current_run and s.end - current_run[0].start >= 15:
            best_run = current_run
            break
        elif s.to_synth or not s.is_speech or (current_run and s.start - current_run[-1].end >= gap):
            if current_run:
                runs.append(current_run)
                current_run = []
        else:
            current_run.append(s)
    if current_run:
        runs.append(current_run)

    # Find the longest run
    if best_run is None:
        best_run = max(runs, key=lambda run: run[-1].end - run[0].start, default=None)
        if not best_run:
            raise HTTPException(400, "No clean speech segments for voice print")

    start = best_run[0].start
    end = best_run[-1].end

    transcription = " ".join(s.text for s in best_run)
    transcription = normalize_text(transcription)

    out = os.path.join(gettempdir(), f"{uuid.uuid4().hex}_vp.wav")
    (
        ffmpeg
        .input(orig_path)
        .filter('atrim', start=start, end=end)
        .filter('asetpts', 'PTS-STARTPTS')
        .output(out, acodec="pcm_s16le", ar=16000)
        .run(overwrite_output=True)
    )

    return out, transcription # local path and transcription text


def synth_segments_word_bleed(wordTokens: List[WordToken], index: int):
    """
    Adjust word tokens to synthesize whole sentences without overlap.
    If a word is marked for synthesis, its neighbors are adjusted to avoid overlap.
    Bleed range is 3 word on either side or until a punctuation mark is found.
    """
    continue_left, continue_right = True, True
    for i in range(1, 3):

        continue_left = (continue_left
                         and index - i >= 0
                         and wordTokens[index - i].is_speech
                         and wordTokens[index - i].text[-1] not in ".!?,;:")

        continue_right = (continue_right
                          and index + i < len(wordTokens)
                          and wordTokens[index + i].is_speech
                          and wordTokens[index + i - 1].text[-1] not in ".!?,;:")

        if continue_left:
            wordTokens[index].text = wordTokens[index - i].text + " " + wordTokens[index].text
            wordTokens[index - i].is_speech = False

        if continue_right:
            wordTokens[index].text += " " + wordTokens[index + i].text
            wordTokens[index + i].is_speech = False

    wordTokens[index].text = normalize_text(wordTokens[index].text)

def synth_segments(vp_path: str, wordTokens: List[WordToken], transcription: str):
    """Send each bad clip’s text + voice-print file to TTS, save returned audio."""

    for i, s in enumerate(wordTokens):
        if s.to_synth and s.is_speech:
            # Adjust neighbors to avoid overlap
            synth_segments_word_bleed(wordTokens, i)
            with open(vp_path, "rb") as vp:
                files = {
                    "audio_file": ("vp.wav", vp, "audio/wav")
                }
                data = {
                        "input_transcription": transcription,
                        "target_text": s.text,
                        'top_p': '0.95',           #  Adjusts the diversity of generated content
                        'temperature': '0.7'    #  Controls randomness in output
                }
                resp = requests.post(
                    "http://localhost:9000/generate-speech",
                    data=data, files=files
                )


            resp.raise_for_status()
            # assume raw audio bytes returned
            ct = resp.headers.get("Content-Type", "")
            ext = ".wav" if "wav" in ct else ".mp3"
            out = os.path.join(gettempdir(), uuid.uuid4().hex + ext)
            with open(out, "wb") as f:
                f.write(resp.content)
            s.synth_path = out

def stitch_all(orig_path: str, wordTokens: List[WordToken]) -> str:
    """Sequentially concat original & synth clips with 20 ms crossfade."""
    # build list of ffmpeg inputs
    clips = []
    prev_end = 0
    for s in wordTokens:
        if s.synth_path and s.to_synth:
            clips.append(ffmpeg.input(s.synth_path))
        elif s.is_speech and not s.to_synth:
            clip = (
                ffmpeg
                .input(orig_path)
                .filter("atrim", start=prev_end, end=s.end)
                .filter("asetpts", "PTS-STARTPTS")
            )
            clips.append(clip)
        prev_end = s.end + 0.001  # retain speech flow

    # sequential 20 ms crossfade
    cur = clips[0]
    for nxt in clips[1:]:
        cur = ffmpeg.filter(
            [cur, nxt], "acrossfade",
            d=0.02,    # duration = 20ms
            c1="tri",  # fade curve
            c2="tri"
        )

    out_path = os.path.join(gettempdir(), "gen.wav")
    ffmpeg.output(cur, out_path).run(overwrite_output=True)
    return out_path  # local path

# ——— Endpoint ————————————————————————————————————————————————————���—————

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Stitcher service is running"
    }

@app.post("/fix-audio", response_model=FixResponse)
async def fix_audio(
    file: UploadFile = File(...),
    payload: str = Form(...)
):
    try:
        # Parse JSON string to list of WordToken objects
        from .models.stitcherModels import wordtokens_from_json
        word_tokens = wordtokens_from_json(payload)


        orig_path = await save_upload_file(file)
        vp_path, transcription = make_voice_print(orig_path, word_tokens)
        synth_segments(vp_path, word_tokens, transcription)
        result = stitch_all(orig_path, word_tokens)
        return FixResponse(fixed_url=f"file://{result}")

    except ValueError as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid payload: {str(e)}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
