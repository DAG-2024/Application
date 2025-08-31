import numpy as np
import librosa

def calculate_adaptive_thresholds(y, sr, frame_length=2048, hop_length=1024,
                                weak_percentile=10, noise_percentile=91,
                                min_weak_thresh_db=-50.0, max_noise_thresh_db=-10.0,
                                min_separation=20.0):
    """
    Calculate adaptive thresholds based on the audio's energy distribution.

    Args:
        y: mono audio signal
        sr: sample rate (Hz)
        frame_length: window size in samples
        hop_length: step size between windows (samples)
        weak_percentile: percentile to use for weak threshold (default: 20)
        noise_percentile: percentile to use for noise threshold (default: 90)
        min_weak_thresh_db: minimum weak threshold (dB)
        max_noise_thresh_db: maximum noise threshold (dB)

    Returns:
        weak_thresh_db, noise_thresh_db: adaptive threshold values
    """

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    rms_db = rms_db[np.isfinite(rms_db)]  # filter out -inf

    if len(rms_db) == 0:
        return -40.0, -15.0

    weak_thresh_db  = np.percentile(rms_db, weak_percentile)
    noise_thresh_db = np.percentile(rms_db, noise_percentile)

    weak_thresh_db  = max(weak_thresh_db, min_weak_thresh_db)
    noise_thresh_db = min(noise_thresh_db, max_noise_thresh_db)

    # enforce minimum separation
    if noise_thresh_db - weak_thresh_db < min_separation:
        mid = (noise_thresh_db + weak_thresh_db) / 2
        weak_thresh_db  = mid - min_separation / 2
        noise_thresh_db = mid + min_separation / 2

    return weak_thresh_db, noise_thresh_db


def detect_energy_segments(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 1024,
    weak_thresh_db: float = None,
    noise_thresh_db: float = None,
    adaptive: bool = True,
    min_segment_duration: float = 0.2,  # drop/merge blips shorter than this
    apply_smoothing: bool = True,  # median filter per-frame dB
    max_merge_gap: float = 0.1  # MERGE: same-label segments separated by <= this gap (sec)
):
    """
    Segment audio by energy levels: weak, normal, loud, with smoothing, min duration,
    and merging of adjacent same-label segments across tiny gaps.

    Args:
      y                 : mono audio signal
      sr                : sample rate (Hz)
      frame_length      : window size in samples
      hop_length        : step size between windows (samples)
      weak_thresh_db    : below -> 'weak_activity' (if None and adaptive=True, calculated automatically)
      noise_thresh_db   : above -> 'loud_noise' (if None and adaptive=True, calculated automatically)
      adaptive          : whether to use adaptive thresholds
    Returns:
      List of segments, each with:
        - start_time, end_time (sec, rounded to 2dp)
        - label / {'weak_activity','normal_speech','loud_noise'}
        - mean_db    : average dB over the segment
        - score      : normalized energy [0...1]
    """

    # thresholds
    if adaptive or weak_thresh_db is None or noise_thresh_db is None:
        weak_thresh_db, noise_thresh_db = calculate_adaptive_thresholds(y, sr, frame_length, hop_length)
    else:
        weak_thresh_db = -40.0 if weak_thresh_db is None else weak_thresh_db
        noise_thresh_db = -15.0 if noise_thresh_db is None else noise_thresh_db

    # frame energies
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    times = librosa.frames_to_time(
        np.arange(len(rms_db)),
        sr=sr,
        hop_length=hop_length,
        n_fft=frame_length
    )

    # optional smoothing to reduce jitter
    if apply_smoothing and len(rms_db) > 5:
        from scipy.signal import medfilt
        rms_db = medfilt(rms_db, kernel_size=5)

    # helpers
    def label_db(db):
        if db < weak_thresh_db:  return "weak_activity"
        if db > noise_thresh_db: return "loud_noise"
        return "normal_speech"

    def compute_score(mean_db):
        raw = (mean_db - weak_thresh_db) / (noise_thresh_db - weak_thresh_db)
        return float(np.clip(raw, 0.0, 1.0))

    # segments hold provisional fields: start, end, label, sum_db, count
    segments = []
    curr = None  # current run: start, end, label, energies

    def flush_and_maybe_merge(run):
        """Flush current run if long enough; merge into previous segment if same label and tiny gap."""
        if run is None:
            return
        duration = run["end"] - run["start"]
        if duration < min_segment_duration:
            return  # drop micro-blip

        new_seg = {
            "start": run["start"],
            "end": run["end"],
            "label": run["label"],
            "sum_db": float(np.sum(run["energies"])),
            "count": len(run["energies"])
        }

        # Try to merge with previous if same label and small gap
        if segments and segments[-1]["label"] == new_seg["label"]:
            gap = new_seg["start"] - segments[-1]["end"]
            if gap <= max_merge_gap:
                # merge by extending end and summing stats
                segments[-1]["end"] = new_seg["end"]
                segments[-1]["sum_db"] += new_seg["sum_db"]
                segments[-1]["count"] += new_seg["count"]
                return

        # otherwise start a new segment
        segments.append(new_seg)

    # main loop: run-length accumulate by label
    for t, db in zip(times, rms_db):
        lab = label_db(db)
        if curr is None or lab != curr["label"]:
            # label change -> flush previous (with merge logic)
            flush_and_maybe_merge(curr)
            # start new run
            curr = {
                "start": t,
                "end": t + hop_length / sr,
                "label": lab,
                "energies": [db]
            }
        else:
            curr["end"] = t + hop_length / sr
            curr["energies"].append(db)

    # flush last run
    flush_and_maybe_merge(curr)

    # finalize: compute mean_db + score and round times
    finalized = []
    for seg in segments:
        mean_db = seg["sum_db"] / max(seg["count"], 1)
        finalized.append({
            "start_time": round(seg["start"], 2),
            "end_time": round(seg["end"], 2),
            "label": seg["label"],
            "mean_db": float(mean_db),
            "score": compute_score(float(mean_db))
        })

    return finalized


def detect_energy(audio_path, adaptive=True, **kwargs):
    """
    High-level wrapper: load file, run detection.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    return detect_energy_segments(y, sr, adaptive=adaptive, **kwargs)