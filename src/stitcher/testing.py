import models.stitcherModels as stitcherModels
from fastapi.testclient import TestClient
from stitcher import app

if __name__ == "__main__":
    wordtokens = stitcherModels.load_wordtokens_json("testing/audio2.json")
    payload_json = stitcherModels.wordtokens_to_json(wordtokens)

    client = TestClient(app)
    with open("testing/sample_audio2.wav", "rb") as audio_file:
        response = client.post(
            "/fix-audio",
            data={"payload": payload_json},
            files={"file": ("audio.wav", audio_file, "audio/wav")},
        )
        print("Status code:", response.status_code)
        print("Response:", response.json())

        if response.status_code == 200:
            fixed_url = response.json().get("fixed_url")
            if fixed_url and fixed_url.startswith("file://"):
                file_path = fixed_url[7:]
                with open(file_path, "rb") as src, open("testing/output.wav", "wb") as dst:
                    dst.write(src.read())
                print("Saved output to output.wav")
            else:
                print("No valid fixed_url in response.")
        else:
            print("Request failed.")