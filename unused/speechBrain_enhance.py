from speechbrain.inference.enhancement import WaveformEnhancement
import torchaudio

def SB_enhance(in_path, out_path):
    enhance_model = WaveformEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir="pretrained_models/mtl-mimic-voicebank",
    )
    enhanced = enhance_model.enhance_file(in_path)

    # Saving enhanced signal on disk
    torchaudio.save(out_path, enhanced.unsqueeze(0).cpu(), 16000)


