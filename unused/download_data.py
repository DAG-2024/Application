from datasets import load_dataset
import os
import soundfile as sf

# Load ~500 English samples
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train[:500]")

output_dir = "common_voice_samples"
os.makedirs(output_dir, exist_ok=True)

for i, sample in enumerate(dataset):
    audio = sample["audio"]
    path = os.path.join(output_dir, f"sample_{i}.wav")
    sf.write(path, audio["array"], audio["sampling_rate"])