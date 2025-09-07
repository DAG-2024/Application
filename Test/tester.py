import requests
import json
from pathlib import Path
from typing import List, Optional
import logging
import os

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("tester")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(log_dir, "tester.log"))
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


CONTROLLER_URL = "0.0.0.0"
CONTROLLER_PORT = 9002

STITCHER_URL = "0.0.0.0"
STITCHER_PORT = 9001

# --- CONFIG ---
FEED_AUDIO_URL = f"http://{CONTROLLER_URL}:{CONTROLLER_PORT}/feed-audio"
FIX_AUDIO_URL =  f"http://{STITCHER_URL}:{STITCHER_PORT}/fix-audio"
AUDIO_FILE_PATH = "sample_audio.wav"
OUTPUT_WAV_PATH = "fixed_output.wav"
DATASET_DIR = "../../noisy"
OUTPUT_DIR = "output"
FILE_PATTERN = "*.wav"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset_path = Path(DATASET_DIR)
    audio_files = list(dataset_path.glob(FILE_PATTERN))
    if not audio_files:
        logger.warning(f"No audio files found in {DATASET_DIR} with pattern {FILE_PATTERN}")
        return
    logger.info(f"Found {len(audio_files)} audio files to process")

    results = []

    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file.name}")
        audio_path = DATASET_DIR + '/' + audio_file.name
        try:
            with open(audio_path, "rb") as f:
                # 1. Call feed_audio endpoint
                files = {"file": ("audio.wav", f, "audio/wav")}
                response = requests.post(FEED_AUDIO_URL, files=files)
                response.raise_for_status()
                result = response.json()
                wordtokens_json = result["wordtokens"]


            transcripts = " ".join([t['text'] for t in wordtokens_json]).encode('utf-8')

            with open(audio_path, "rb") as f:
                # 1. Call feed_audio endpoint
                files = {"file": ("audio.wav", f, "audio/wav")}
                # Ensure wordtokens_json is a string
                payload_str = wordtokens_json if isinstance(wordtokens_json, str) else json.dumps(wordtokens_json)
                data = {"payload": payload_str}
                params = {"balance": False}
                response = requests.post(
                    FIX_AUDIO_URL,
                    files=files,
                    data=data,
                    params=params
                )
                response.raise_for_status()
                result = response.json()
                print(response.text)
                fixed_url = result["fixed_url"]

        except Exception as e:
            print(f"Error during API calls: {e}")
            exit(1)

        # 3. Download the fixed audio file
        if fixed_url.startswith("file://"):
            local_path = fixed_url[7:]
            with open(local_path, "rb") as f_in, open(OUTPUT_DIR + '/' + audio_file.name, "wb") as f_out:
                f_out.write(f_in.read())
            with open(OUTPUT_DIR + '/' + audio_file.name + '.txt', "wb") as f:
                f.write(transcripts)


if __name__ == "__main__":
    main()