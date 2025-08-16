from speechbrain.inference.enhancement import WaveformEnhancement
import torchaudio

def main():
    enhance_model = WaveformEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir="pretrained_models/mtl-mimic-voicebank",
    )
    enhanced = enhance_model.enhance_file("/Users/david/DAG-project/Word_Detection/input/SpEAR-speech-database-master/data/Noisy_Recordings/bigtips_factoryr1_16.wav")

    # Saving enhanced signal on disk
    torchaudio.save('enhanced.wav', enhanced.unsqueeze(0).cpu(), 16000)



if __name__ == "__main__":
    main()
