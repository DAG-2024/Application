from typing import List, Tuple
import numpy as np
import librosa
import webrtcvad

# =========================
# Core utilities
# =========================

def frame_indices(n_samples: int, sr: int, frame_ms: int = 20, hop_ms: int = 10):
    """
    Create index windows for framing the audio signal.
    """
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    idx = [(i, min(i + frame, n_samples)) for i in range(0, n_samples - frame + 1, hop)]
    return idx, frame, hop


def to_pcm16(y: np.ndarray) -> bytes:
    """
    Convert floating-point waveform (-1.0..1.0) to 16-bit PCM bytes for WebRTC VAD.
    """
    y16 = np.clip((y * 32768.0).astype(np.int16), -32768, 32767)
    return y16.tobytes()


def webrtc_speech_mask(y: np.ndarray, sr: int, mode: int = 2, frame_ms: int = 20, hop_ms: int = 10):
    """
    WebRTC VAD: return a boolean mask per hop where True=Speech.
    mode: 0 (lenient) .. 3 (strict). Higher means fewer false speech detections.
    frame_ms: frame size in ms (10, 20, or 30), meaning the VAD decision is made every frame_ms.
    hop_ms: hop size in ms (usually ≤ frame_ms, e.g. 10ms)
    """
    assert sr in (8000, 16000, 32000, 48000), "WebRTC VAD requires 8/16/32/48 kHz"
    vad = webrtcvad.Vad(mode)
    idx, frame, hop = frame_indices(len(y), sr, frame_ms, hop_ms)
    pcm = to_pcm16(y)
    speech = []
    for (s, e) in idx:
        chunk = pcm[2 * s:2 * e]  # 2 bytes per sample
        speech.append(vad.is_speech(chunk, sr))

    return np.array(speech, dtype=bool), idx, hop


# =========================
# Acoustic features & events
# =========================

def extract_acoustic_features(y: np.ndarray, sr: int, frame_len: int = 2048, hop_len: int = 512):
    """
    Compute frame-level acoustic features used to detect artifacts/noise:
      - RMS (dB): loudness
      - Spectral flatness: noise-like vs. tonal
      - Spectral flux: sudden spectral change (transients)
      - Zero-crossing rate: crackle/hiss/high-frequency cues
    """
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_len, win_length=frame_len)) + 1e-10

    rms = librosa.feature.rms(S=S).squeeze()
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    flatness = librosa.feature.spectral_flatness(S=S).squeeze()

    # Normalize columns to compare shapes, then compute L2 flux across time
    S_norm = S / np.maximum(S.sum(axis=0, keepdims=True), 1e-12)
    flux = np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0))
    flux = np.concatenate([[flux[0]], flux])  # pad first frame

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len).squeeze()

    return rms_db, flatness, flux, zcr


def mask_to_segments(mask: np.ndarray, hop_s: float) -> List[Tuple[float, float]]:
    """
    Convert a boolean frame mask to merged time segments.
    """
    n = len(mask)
    # Find contiguous runs
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    # Filter by min length and convert to seconds
    out: List[Tuple[float, float]] = []
    for s, e in runs:
        out.append((s * hop_s, e * hop_s))
    return out


def detect_acoustic_events_in_speech(
    y: np.ndarray,
    sr: int,
    hop_len: int,
    rms_db_boost: float = 6.0,  # how much louder than the background a frame must be to count
    flat_th: float = 0.35,   # higher -> only very noisy textures are marked
    flux_z: float = 1.5,    # controls sensitivity to sudden changes
    zcr_th: float = 0.2,  # controls detection of crackles / high-frequency artifacts
) -> List[Tuple[float, float]]:
    """
    Detect artifact/noise events *within* speech regions by combining features.
    """
    rms_db, flatness, flux, zcr = extract_acoustic_features(y, sr, frame_len=2048, hop_len=hop_len)
    L = min(len(rms_db), len(flatness), len(flux), len(zcr))
    rms_db, flatness, flux, zcr = rms_db[:L], flatness[:L], flux[:L], zcr[:L]

    noise_floor = np.percentile(rms_db, 15)
    loud = rms_db > (noise_floor + rms_db_boost)

    flux_zscore = (flux - flux.mean()) / (flux.std() + 1e-8)

    # A frame is a "defect candidate" if any abnormal feature triggers.
    defect = (loud & (flatness > flat_th)) | (flux_zscore > flux_z) | (zcr > zcr_th)

    hop_s = hop_len / sr
    acoustic_events = mask_to_segments(defect, hop_s)
    return acoustic_events

# =========================
# End-to-end function
# =========================

def analyze_recording(
    wav_path: str,
    # acoustic thresholds
    rms_db_boost: float = 6.0,  # how much louder than the background a frame must be to count as “loud.”
    flat_th: float = 0.35,      # higher → only very noisy textures are marked.
    flux_z: float = 1.5,        # controls sensitivity to sudden changes.
    zcr_th: float = 0.2,        # controls detection of crackles / high-frequency artifacts.
    hop_ms: int = 10,           # hop size in ms

) -> List[Tuple[float, float]]:
    """
    End-to-end analysis:
      - Load audio
      - Acoustic defects within speech
    """

    # 1) Load audio mono @16k (required for WebRTC VAD)
    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    hop_len = int(sr * hop_ms / 1000)

    # 2) Acoustic defect events restricted to speech
    events = detect_acoustic_events_in_speech(
        y, sr, hop_len,
        rms_db_boost=rms_db_boost,
        flat_th=flat_th,
        flux_z=flux_z,
        zcr_th=zcr_th,
    )

    return events