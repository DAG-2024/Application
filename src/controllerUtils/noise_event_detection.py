from typing import List, Tuple, Dict
import numpy as np
import librosa
import webrtcvad
from dataclasses import dataclass

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


# =========================
# Helpers
# =========================

def segments_to_mask(segments: List[Tuple[float, float]], length: int, hop_s: float) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for s, e in segments:
        i0 = max(0, int(np.floor(s / hop_s)))
        i1 = min(length, int(np.ceil(e / hop_s)))
        if i1 > i0:
            mask[i0:i1] = True
    return mask

def merge_and_filter_segments(segments: List[Tuple[float, float]], min_dur: float, min_gap: float) -> List[Tuple[float, float]]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    cs, ce = segments[0]
    for s, e in segments[1:]:
        if s - ce <= min_gap:
            ce = max(ce, e)
        else:
            if (ce - cs) >= min_dur:
                merged.append((cs, ce))
            cs, ce = s, e
    if (ce - cs) >= min_dur:
        merged.append((cs, ce))
    return merged

@dataclass
class _FeatPack:
    rms_db: np.ndarray
    flat: np.ndarray
    flux_z: np.ndarray
    zcr: np.ndarray
    noise_floor: float

def _compute_features_and_masks(y: np.ndarray, sr: int, hop_ms: int, vad_mode: int) -> tuple[_FeatPack, np.ndarray, float]:
    hop_len = int(sr * hop_ms / 1000)
    rms_db, flatness, flux, zcr = extract_acoustic_features(y, sr, frame_len=2048, hop_len=hop_len)
    L = min(len(rms_db), len(flatness), len(flux), len(zcr))
    rms_db, flatness, flux, zcr = rms_db[:L], flatness[:L], flux[:L], zcr[:L]
    hop_s = hop_len / sr

    # Adaptive noise floor from the lower percentile of RMS
    noise_floor = np.percentile(rms_db, 15)

    # Z-score flux per recording
    flux_z = (flux - flux.mean()) / (flux.std() + 1e-8)

    # Speech gating via WebRTC VAD aligned to feature frames
    speech_mask_vad, vad_idx, vad_hop = webrtc_speech_mask(y, sr, mode=vad_mode, frame_ms=20, hop_ms=hop_ms)
    vad_hop_s = vad_hop / sr
    speech_segments = mask_to_segments(speech_mask_vad, vad_hop_s)
    speech_mask_feat = segments_to_mask(speech_segments, L, hop_s)

    feats = _FeatPack(rms_db=rms_db, flat=flatness, flux_z=flux_z, zcr=zcr, noise_floor=noise_floor)
    return feats, speech_mask_feat, hop_s


def _classify_segment(i0: int, i1: int, feats: _FeatPack, hop_s: float) -> tuple[str, float]:
    dur_s = (i1 - i0) * hop_s
    f_rms = feats.rms_db[i0:i1]
    f_flat = feats.flat[i0:i1]
    f_fluxz = feats.flux_z[i0:i1]
    f_zcr = feats.zcr[i0:i1]

    rms_boost = float(np.maximum(0.0, f_rms.mean() - feats.noise_floor))
    flat_m = float(f_flat.mean())
    flux_m = float(f_fluxz.mean())
    zcr_m = float(f_zcr.mean())

    # Simple heuristics
    if dur_s < 0.08 and flux_m > 2.5:
        label = 'click'
    elif flat_m > 0.6 and zcr_m > 0.25 and dur_s >= 0.15:
        label = 'hiss'
    elif flat_m < 0.2 and zcr_m < 0.1 and dur_s >= 0.15:
        label = 'hum'
    else:
        label = 'loud_noise'

    # Severity score in [0, 1] via squashed weighted sum
    raw = 0.6 * (rms_boost / 6.0) + 0.5 * flux_m + 0.4 * (flat_m - 0.35) + 0.2 * (zcr_m - 0.2)
    score = 1.0 / (1.0 + np.exp(-raw))
    score = float(np.clip(score, 0.0, 1.0))
    return label, score

# =========================
# Improved detector (new)
# =========================

def detect_noise_events(
    wav_path: str,
    hop_ms: int = 10,
    vad_mode: int = 2,
    rms_db_boost: float = 6.0,
    flat_th: float = 0.35,
    flux_z: float = 1.5,
    zcr_th: float = 0.2,
    min_event_dur: float = 0.05,
    min_gap: float = 0.05,
    speech_overlap_th: float = 0.3,  # fraction of frames overlapped with VAD speech to count as masking
) -> List[Dict[str, float | str | bool]]:
    """
    Detect loud/noisy events across the whole file, then tag whether each event overlaps speech.
    Returns dicts with keys:
      start, end, label='loud_noise', score, label_detail, masking_speech (bool), speech_overlap (0..1)
    """
    # 16 kHz mono for VAD stability
    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    # Features + speech mask aligned to feature frames
    feats, speech_mask_feat, hop_s = _compute_features_and_masks(y, sr, hop_ms, vad_mode)

    # Core defect mask (no VAD gating here so we also get non‑speech noises)
    loud = feats.rms_db > (feats.noise_floor + rms_db_boost)
    defect = (loud & (feats.flat > flat_th)) | (feats.flux_z > flux_z) | (feats.zcr > zcr_th)

    # Segment smoothing
    segs = mask_to_segments(defect, hop_s)
    segs = merge_and_filter_segments(segs, min_event_dur, min_gap)

    # Classify, score, and compute speech overlap tag
    events: List[Dict[str, float | str | bool]] = []
    for s, e in segs:
        i0 = max(0, int(np.floor(s / hop_s)))
        i1 = max(i0 + 1, int(np.ceil(e / hop_s)))

        # Speech overlap ratio for this event
        total = max(1, i1 - i0)
        sp_frames = int(np.sum(speech_mask_feat[i0:i1]))
        sp_overlap = float(sp_frames / total)
        masking = bool(sp_overlap >= speech_overlap_th)

        # Acoustic class and severity
        label_detail, score = _classify_segment(i0, i1, feats, hop_s)

        events.append({
            'start': float(s),
            'end': float(e),
            'label': 'loud_noise',           # keep downstream compatibility
            'label_detail': label_detail,    # click/hiss/hum/loud_noise
            'score': float(score),
            'masking_speech': masking,       # True => loud noise overlapping speech
            'speech_overlap': sp_overlap,    # 0..1 fraction
        })

    return events
