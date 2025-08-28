"""
Defect noise event for single-speaker speech recordings
==================================================

This module detects problematic audio segments and fuses them with Whisper's
word-level confidences to decide whether to:
  - RESYNTHESIZE (replace speech),
  - NOISE_SUPPRESS (keep speech, suppress noise),
  - MUMBLE_RESYNTHESIZE (replace likely mumbled speech).

Pipeline:
1) WebRTC VAD -> speech gating
2) Acoustic features -> transient/noise event detection
3) Load Whisper transcript (word-level timestamps + confidences)
4) Fuse acoustic events + Whisper confidences
5) Output actionable segments with timestamps & decision labels

Author: you :)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import json
import numpy as np
import librosa
import webrtcvad



# =========================
# Data structures
# =========================

@dataclass
class Word:
    """Minimal word representation from Whisper output."""
    text: str
    start: float
    end: float
    confidence: Optional[float]  # Some Whisper builds may not produce confidences


@dataclass
class ProblemSegment:
    """Acoustic event segment (likely noise/artifact) in seconds."""
    start: float
    end: float
    score: float = 0.0  # optional intensity score if you compute one


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


def mask_to_segments(mask: np.ndarray, hop_s: float, min_ms: int = 30, merge_gap_ms: int = 120) -> List[ProblemSegment]:
    """
    Convert a boolean frame mask to merged time segments.
    - min_ms: drop very short blips
    - merge_gap_ms: join segments separated by short gaps
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

    # Merge close runs
    merged: List[Tuple[int, int]] = []
    gap_hops = int(round(merge_gap_ms / 1000.0 / hop_s))
    for r in runs:
        if not merged:
            merged.append(r)
            continue
        ps, pe = merged[-1]
        cs, ce = r
        if cs - pe <= gap_hops:
            merged[-1] = (ps, ce)
        else:
            merged.append(r)

    # Filter by min length and convert to seconds
    out: List[ProblemSegment] = []
    min_hops = int(round(min_ms / 1000.0 / hop_s))
    for s, e in merged:
        if (e - s) >= min_hops:
            out.append(ProblemSegment(start=s * hop_s, end=e * hop_s, score=0.0))
    return out


def detect_acoustic_events(
    y: np.ndarray,
    sr: int,
    hop_len: int,
    rms_db_boost: float = 6.0,
    flat_th: float = 0.35,
    flux_z: float = 1.5,
    zcr_th: float = 0.2,
    min_ms: int = 30,
    merge_gap_ms: int = 120,
) -> List[ProblemSegment]:
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
    problem_mask = (loud & (flatness > flat_th)) | (flux_zscore > flux_z) | (zcr > zcr_th)

    hop_s = hop_len / sr
    return mask_to_segments(problem_mask, hop_s, min_ms=min_ms, merge_gap_ms=merge_gap_ms)

# =========================
# Fusion logic
# =========================

def interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Return the length (in seconds) of overlap between intervals a=[a0,a1], b=[b0,b1].
    """
    a0, a1 = a
    b0, b1 = b
    return max(0.0, min(a1, b1) - max(a0, b0))


def label_words_by_fusion(
    words: List[Word],
    acoustic_events: List[ProblemSegment],
    low_conf_th: float = 0.58,            # words below this are suspicious
    min_overlap_for_effect: float = 0.02, # at least 20 ms overlap to count (avoid 1-frame jitters)
) -> List[Tuple[Word, str, float]]:
    """
    For each word, compute whether it overlaps an acoustic defect and choose a per-word label:
      - RESYNTHESIZE if low confidence + overlaps defect
      - NOISE_SUPPRESS if good confidence + overlaps defect
      - MUMBLE_RESYNTHESIZE if low confidence + no defect
      - OK otherwise

    Returns:
      List of (Word, label, overlap_seconds)
    """
    labeled = []
    events = [(e.start, e.end) for e in acoustic_events]

    for w in words:
        w_iv = (w.start, w.end)
        ov = 0.0
        for e in events:
            ov += interval_overlap(w_iv, e)
        # Small epsilon to ignore spurious micro-overlaps
        has_defect = ov >= min_overlap_for_effect

        # Confidence may be None (fallback Whisper) -> treat as "unknown"
        conf = w.confidence
        low_conf = (conf is not None) and (conf < low_conf_th)

        if low_conf and has_defect:
            label = "RESYNTHESIZE"
        elif (conf is not None) and (not low_conf) and has_defect:
            label = "NOISE_SUPPRESS"
        elif low_conf and not has_defect:
            label = "MUMBLE_RESYNTHESIZE"
        else:
            label = "OK"

        labeled.append((w, label, ov))
    return labeled


# =========================
# End-to-end function
# =========================

def analyze_recording(
    wav_path: str,
    vad_mode: int = 2,          # 0 permissive (may let background in), 3 strict (only clean speech).
    # acoustic thresholds
    rms_db_boost: float = 6.0,  # how much louder than the background a frame must be to count as “loud.”
    flat_th: float = 0.35,      # higher → only very noisy textures are marked.
    flux_z: float = 1.5,        # controls sensitivity to sudden changes.
    zcr_th: float = 0.2,        # controls detection of crackles / high-frequency artifacts.
    min_ms=30,              # minimum duration of an acoustic event to keep (ms)
    merge_gap_ms=120         # join acoustic events separated by ≤ this gap (ms)

)  -> Tuple[List[ProblemSegment], List[ProblemSegment]]:

    """
    End-to-end analysis:
      - Load audio
      - VAD -> speech mask
      - Acoustic defects within speech
      - Load Whisper words
      - Label words and produce final actionable spans

    Returns:
      (acoustic_events, action_spans)
    """

    # 1) Load audio mono @16k (required for WebRTC VAD)
    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    # 2) Speech mask via VAD
    vad_mask, _, hop_len = webrtc_speech_mask(y, sr, mode=vad_mode)

    hop_s = hop_len / sr
    vad_events = mask_to_segments(vad_mask, hop_s, min_ms=min_ms, merge_gap_ms=merge_gap_ms)

    # 3) Acoustic defect events restricted to speech
    acoustic_events = detect_acoustic_events(
        y, sr, hop_len,
        rms_db_boost=rms_db_boost,
        flat_th=flat_th,
        flux_z=flux_z,
        zcr_th=zcr_th,
        min_ms=min_ms,             # keep short min here; we'll shape spans later
        merge_gap_ms=merge_gap_ms
    )

    return acoustic_events, vad_events