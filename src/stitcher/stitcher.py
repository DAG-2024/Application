from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from models.stitcherModels import WordToken, FixResponse
from typing import List
import uuid, os, requests
import ffmpeg
from tempfile import gettempdir

app = FastAPI()


async def save_upload_file(upload: UploadFile) -> str:
    """Save uploaded UploadFile to a tmp file and return its path."""
    ext = os.path.splitext(upload.filename)[1] or ".wav"
    path = os.path.join(gettempdir(), uuid.uuid4().hex + ext)
    data = await upload.read()
    with open(path, "wb") as f:
        f.write(data)
    return path

def make_voice_print(orig_path: str, WordTokens: List[WordToken]) -> str:
    """
    Extract the longest sequential clean speech segment (up to 15s) from the original audio using WordToken timings.
    """
    # Find all runs of clean speech
    runs = []
    current_run = []
    for s in WordTokens:
        if not s.to_synth and s.is_speech:
            current_run.append(s)
        else:
            if current_run:
                runs.append(current_run)
                current_run = []
    if current_run:
        runs.append(current_run)
    # Find the longest run
    best_run = max(runs, key=lambda run: run[-1].end - run[0].start, default=None)
    if not best_run:
        raise HTTPException(400, "No clean speech segments for voice print")
    # Trim to max 15 seconds
    start = best_run[0].start
    end = best_run[-1].end
    if end - start > 15:
        end = start + 15
    out = os.path.join(gettempdir(), f"{uuid.uuid4().hex}_vp.wav")
    (
        ffmpeg
        .input(orig_path)
        .filter('atrim', start=start, end=end)
        .filter('asetpts', 'PTS-STARTPTS')
        .output(out, acodec="pcm_s16le")
        .run(overwrite_output=True)
    )
    return out  # local path

def synth_segments(vp_path: str, wordTokens: List[WordToken]):
    """Send each bad clip’s text + voice-print file to TTS, save returned audio."""
    # For testing, assume transcription is available as a string
    vp_transcription = "things escalate a bit. So how do you represent letters? Because obviously this makes our devices more useful, whether it's in English or any other human language. How could we go about representing the letter A for instance?."
    # Trim vp_path to max 15 seconds for each request
    trimmed_vp_path = os.path.join(gettempdir(), f"{uuid.uuid4().hex}_vp15s.wav")
    (
        ffmpeg
        .input(vp_path, t=15)
        .output(trimmed_vp_path, acodec="pcm_s16le")
        .run(overwrite_output=True)
    )
    for s in wordTokens:
        if s.to_synth and s.is_speech:
            with open(vp_path, "rb") as vp:
                files = {
                    "audio_file": ("vp.wav", vp, "audio/wav")
                }
                data = {"input_transcription": vp_transcription,"target_text": s.text}
                resp = requests.post(
                    "https://localhost:9000/generate-speech",
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
    for s in wordTokens:
        if s.synth_path:
            clips.append(ffmpeg.input(s.synth_path))
        else:
            clip = (
                ffmpeg
                .input(orig_path)
                .filter("atrim", start=s.start, end=s.end)
                .filter("asetpts", "PTS-STARTPTS")
            )
            clips.append(clip)

    # sequential 20 ms crossfade
    cur = clips[0]
    for nxt in clips[1:]:
        cur = ffmpeg.filter(
            [cur, nxt], "acrossfade",
            d=0.02,    # duration = 20ms
            c1="tri",  # fade curve
            c2="tri"
        )

    out_path = os.path.join(gettempdir(), f"{uuid.uuid4().hex}_fixed.mp3")
    ffmpeg.output(cur, out_path).run(overwrite_output=True)
    return out_path  # local path

# ——— Endpoint ——————————————————————————————————————————————————————————

@app.post("/fix-audio", response_model=FixResponse)
async def fix_audio(
    file: UploadFile = File(...),
    payload: List[WordToken] = Body(...)
):
    # 1. Save upload locally
    orig_path = await save_upload_file(file)

    # 2. Build voice-print from longest clean speech segment (max 15s)
    vp_path = make_voice_print(orig_path, payload)

    # 3. TTS the bad clips
    synth_segments(vp_path, payload)

    # 4. Stitch everything in sequence with crossfade
    result = stitch_all(orig_path, payload)

    return FixResponse(fixed_url=f"file://{result}")
