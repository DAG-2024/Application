from voicefixer import VoiceFixer
import os

def VF_enhance(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    vf = VoiceFixer()
    vf.restore(
        input=input_path,
        output=output_path,
        mode=1  # Mode 0 = full restoration (speech + background suppression)
    )
    print(f"Enhanced file saved to: {output_path}")

