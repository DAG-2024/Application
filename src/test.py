from controllerUtils.transcriber import transcribe

if __name__ == "__main__":
    AUDIO_FILE_PATH = "sample_audio.wav"
    print(transcribe(AUDIO_FILE_PATH))
