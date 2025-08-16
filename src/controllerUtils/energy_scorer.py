
import numpy as np
import librosa

def detect_energy_segments(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 1024,
    weak_thresh_db: float = -40.0,
    noise_thresh_db: float = -15.0
):
    """
    Args:
      y                 : mono audio signal
      sr                : sample rate (Hz)
      frame_length      : window size in samples
      hop_length        : step size between windows (samples)
      weak_thresh_db    : below -> 'weak_activity'
      noise_thresh_db   : above -> 'loud_noise'
    Returns:
      List of segments, each with:
        - start_time, end_time (sec, rounded to 2dp)
        - label / {'weak_activity','normal_speech','loud_noise'}
        - mean_db    : average dB over the segment
        - score      : normalized energy [0...1]
    """
    # 1) get frame-level RMS -> dB
    rms    = librosa.feature.rms(y=y,
                                  frame_length=frame_length,
                                  hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    times  = librosa.frames_to_time(np.arange(len(rms_db)),
                                    sr=sr,
                                    hop_length=hop_length,
                                    n_fft=frame_length)

    # 2) helpers
    def label_db(db):
        if db < weak_thresh_db:    return "weak_activity"
        if db > noise_thresh_db:   return "loud_noise"
        return "normal_speech"

    def compute_score(mean_db):
        # maps mean_db / [weak_thresh_db, noise_thresh_db] -> [0,1]
        raw = (mean_db - weak_thresh_db) / (noise_thresh_db - weak_thresh_db)
        return float(np.clip(raw, 0.0, 1.0))

    # 3) segment accumulation
    segments = []
    curr = None

    for t, db in zip(times, rms_db):
        lab = label_db(db)
        if curr is None or lab != curr["label"]:
            # flush previous
            if curr is not None:
                mean_db = float(np.mean(curr["energies"]))
                segments.append({
                    "start_time": round(curr["start"], 2),
                    "end_time":   round(curr["end"],   2),
                    "label":      curr["label"],
                    "mean_db":    mean_db,
                    "score":      compute_score(mean_db)
                })
            # start new
            curr = {
                "start":    t,
                "end":      t + hop_length/sr,
                "label":    lab,
                "energies": [db]
            }
        else:
            curr["end"]       = t + hop_length/sr
            curr["energies"].append(db)

    # flush last
    if curr:
        mean_db = float(np.mean(curr["energies"]))
        segments.append({
            "start_time": round(curr["start"], 2),
            "end_time":   round(curr["end"],   2),
            "label":      curr["label"],
            "mean_db":    mean_db,
            "score":      compute_score(mean_db)
        })

    return segments


def detect_energy(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    segs   = detect_energy_segments(y, sr)
    return segs
