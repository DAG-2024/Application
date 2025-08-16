from unused.voiceFixer_enhance import VF_enhance
from unused.speechBrain_enhance import SB_enhance
from openai import OpenAI



def main():

    client = OpenAI(api_key="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")
    audio_file = open('/Users/david/Desktop/מסמכים/DAG project/Audio test samples/HARVARD- CS LECTURE/masked/maked-cs-snippet.wav', "rb")

    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file, 
    response_format="text",
    prompt="""here's a provided audio file of a human speech, which contains a few 
    sections of clear noise, that masks few words in the speech, your job is to transcribe
        the speech, but only the clear and not masked words, dont try to infer the masked 
        words in the speech, instead, detect exactly where the noise is, and insert a {blank} 
        placeholder in the transcribed speec. the output should be nothing but the trnscribed 
        speech, with the placeholders inserted if necessary, as instructed above"""

    )
    a=1

    print(transcription)

if __name__ == "__main__":
    main()
