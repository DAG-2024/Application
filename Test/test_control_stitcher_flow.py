import requests
import json

CONTROLLER_URL = "0.0.0.0"
CONTROLLER_PORT = 9002

STITCHER_URL = "0.0.0.0"
STITCHER_PORT = 9001

# --- CONFIG ---
FEED_AUDIO_URL = f"http://{CONTROLLER_URL}:{CONTROLLER_PORT}/feed-audio"
FIX_AUDIO_URL =  f"http://{STITCHER_URL}:{STITCHER_PORT}/fix-audio"
AUDIO_FILE_PATH = "sample_audio.wav"
OUTPUT_WAV_PATH = "fixed_output.wav"

# 1. Call feed_audio endpoint
with open(AUDIO_FILE_PATH, "rb") as audio_file:
    files = {"file": ("test_audio.wav", audio_file, "audio/wav")}
    response = requests.post(FEED_AUDIO_URL, files=files)
    response.raise_for_status()
    result = response.json()
    wordtokens_json = result["wordtokens"]

# 2. Call fix-audio endpoint
with open(AUDIO_FILE_PATH, "rb") as audio_file:
    files = {"file": ("test_audio.wav", audio_file, "audio/wav")}
    payload = json.loads(wordtokens_json)
    response = requests.post(
        FIX_AUDIO_URL,
        files=files,
        json=payload
    )
    response.raise_for_status()
    result = response.json()
    fixed_url = result["fixed_url"]

# 3. Download the fixed audio file
if fixed_url.startswith("file://"):
    local_path = fixed_url[7:]
    with open(local_path, "rb") as f_in, open(OUTPUT_WAV_PATH, "wb") as f_out:
        f_out.write(f_in.read())
    print(f"Saved fixed audio to {OUTPUT_WAV_PATH}")
else:
    print(f"Unexpected fixed_url: {fixed_url}")
