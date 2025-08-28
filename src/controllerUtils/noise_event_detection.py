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
from typing import List, Dict, Any, Tuple, Optional, Iterable, Set
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


@dataclass
class ActionSpan:
    """
    Fused action span to apply in your editor:
      - label in {"RESYNTHESIZE", "NOISE_SUPPRESS", "MUMBLE_RESYNTHESIZE"}
      - start/end in seconds (with small padding already applied)
      - extra: list of words covered, avg/min confidence, overlap stats
    """
    label: str
    start: float
    end: float
    words: List[Word]
    avg_conf: Optional[float]
    min_conf: Optional[float]
    overlap_ratio: float  # fraction of this span that overlaps acoustic defects


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


def detect_acoustic_events_in_speech(
    y: np.ndarray,
    sr: int,
    vad_mask: np.ndarray,
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
    L = min(len(rms_db), len(vad_mask), len(flatness), len(flux), len(zcr))
    rms_db, flatness, flux, zcr, vad_mask = rms_db[:L], flatness[:L], flux[:L], zcr[:L], vad_mask[:L]

    noise_floor = np.percentile(rms_db, 15)
    loud = rms_db > (noise_floor + rms_db_boost)

    flux_zscore = (flux - flux.mean()) / (flux.std() + 1e-8)

    # A frame is a "defect candidate" if any abnormal feature triggers.
    defect = (loud & (flatness > flat_th)) | (flux_zscore > flux_z) | (zcr > zcr_th)

    # Only consider frames that are inside speech (we care about legibility hits)
    problem_mask = defect & vad_mask

    hop_s = hop_len / sr
    return mask_to_segments(problem_mask, hop_s, min_ms=min_ms, merge_gap_ms=merge_gap_ms)


# =========================
# Whisper JSON ingestion
# =========================

def load_whisper_words(whisper_json: Dict[str, Any]) -> List[Word]:
    """
    Parse Whisper JSON to a list of Word objects.
    Expected formats:
      - whisper.cpp / faster-whisper often expose segment.words with {word, start, end, confidence}
      - openai-whisper may not provide words; then we approximate by splitting by spaces per segment.

    NOTE: If no confidences are present, confidence is set to None (caller must handle).
    """
    words: List[Word] = []

    segments = whisper_json.get("segments", [])
    for seg in segments:
        # Case 1: words present
        if "words" in seg and isinstance(seg["words"], list) and len(seg["words"]) > 0:
            for w in seg["words"]:
                # Some flavors use "word" (string with leading space), others "text".
                text = w.get("word") or w.get("text") or ""
                start = float(w.get("start", seg.get("start", 0.0)))
                end = float(w.get("end", seg.get("end", start)))
                conf = w.get("confidence", None)
                if conf is not None:
                    conf = float(conf)
                words.append(Word(text=text.strip(), start=start, end=end, confidence=conf))
        else:
            # Case 2: no words -> approximate by distributing segment time over whitespace-separated tokens
            seg_text = (seg.get("text") or "").strip()
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            tokens = [t for t in seg_text.split() if t]
            if not tokens or seg_end <= seg_start:
                continue
            dur = seg_end - seg_start
            step = dur / max(1, len(tokens))
            for i, tok in enumerate(tokens):
                w_start = seg_start + i * step
                w_end = min(seg_end, w_start + step)
                # No confidences available -> None
                words.append(Word(text=tok, start=w_start, end=w_end, confidence=None))
    return words

# =========================
# word-masking evaluator
# =========================

def _band_energy_ratio(S_mag, sr, fmin=300, fmax=3400):
    """
    Fraction of energy inside [fmin,fmax] vs. full band for each frame.
    S_mag: |STFT| with shape [freq_bins, frames]
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(S_mag.shape[0]-1)*2)
    band = (freqs >= fmin) & (freqs <= fmax)
    num = (S_mag[band, :]**2).sum(axis=0)
    den = (S_mag**2).sum(axis=0) + 1e-12
    return (num / den).astype(float)

def _pitch_confidence(y, sr, frame_length, hop_length):
    """
    Simple pitch 'confidence': normalized peak of autocorrelation via YIN.
    librosa.yin returns f0; use voiced/unvoiced proxy via f0 > 0.
    """
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=frame_length, hop_length=hop_length)
    # Confidence proxy: voiced=1, unvoiced=0
    conf = (~np.isnan(f0)).astype(float)
    return conf

def evaluate_word_masking(
    y, sr, words, vad_mask, hop_len,
    n_fft=2048, win_len=None,
    # thresholds
    vad_non_speech_frac=0.6,        # ≥60% of frames non-speech inside the word
    low_speech_band_frac=0.35,      # speech-band energy ratio below this is suspicious
    high_flatness=0.5,              # flatness above => noise-like
    very_low_rms_db=-40.0,          # dropouts
    very_high_rms_db=-2.0,          # clipping/saturation relative to max
    low_pitch_conf_frac=0.7,        # ≥70% frames unvoiced
):
    """
    For each word window, compute a masking score ∈ [0,1] by combining cues:
      - VAD says non-speech most of the time
      - Speech-band energy ratio very low
      - Spectral flatness high (noise-like)
      - RMS too low or too close to max (saturated)
      - Pitch confidence low (unvoiced) across the window
    """
    win_len = win_len or n_fft
    # STFT once
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_len, win_length=win_len)) + 1e-10
    rms = librosa.feature.rms(S=S).squeeze()
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    flat = librosa.feature.spectral_flatness(S=S).squeeze()
    band_frac = _band_energy_ratio(S, sr, 300, 3400)
    pitch_conf = _pitch_confidence(y, sr, win_len, hop_len)  # 1=voiced, 0=unvoiced

    hop_s = hop_len / sr
    scores = []

    for w in words:
        f0 = int(np.floor(w.start / hop_s))
        f1 = max(f0+1, int(np.ceil(w.end / hop_s)))
        f0 = max(0, min(f0, len(rms_db)-1))
        f1 = max(1, min(f1, len(rms_db)))

        # Fractions over the word window
        vad_ns_frac   = 1.0 - float(vad_mask[f0:f1].mean()) if (f1>f0) else 1.0
        band_med      = float(np.median(band_frac[f0:f1])) if (f1>f0) else 0.0
        flat_med      = float(np.median(flat[f0:f1])) if (f1>f0) else 0.0
        rms_med_db    = float(np.median(rms_db[f0:f1])) if (f1>f0) else -80.0
        unvoiced_frac = float((1.0 - pitch_conf[f0:f1]).mean()) if (f1>f0) else 1.0

        # Normalize each cue to [0,1] "badness"
        c_vad  = 1.0 if vad_ns_frac >= vad_non_speech_frac else vad_ns_frac / vad_non_speech_frac
        c_band = 1.0 if band_med <= low_speech_band_frac else max(0.0, (low_speech_band_frac - band_med) / low_speech_band_frac)
        c_flat = 1.0 if flat_med >= high_flatness else flat_med / high_flatness
        # rms bad if very low OR (near 0 dBFS relative peak) -> map two tails
        low_tail  = 1.0 if rms_med_db <= very_low_rms_db else max(0.0, (very_low_rms_db - rms_med_db) / abs(very_low_rms_db))
        high_tail = 1.0 if rms_med_db >= very_high_rms_db else max(0.0, (rms_med_db - very_high_rms_db) / 6.0)
        c_rms = max(low_tail, high_tail)
        c_pitch = 1.0 if unvoiced_frac >= low_pitch_conf_frac else unvoiced_frac / low_pitch_conf_frac

        # Combine (weights sum to 1). Boost VAD+band (strong indicators of masking)
        score = 0.30*c_vad + 0.30*c_band + 0.20*c_flat + 0.10*c_rms + 0.10*c_pitch
        scores.append(float(np.clip(score, 0.0, 1.0)))

    return scores


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


def spans_from_labeled_words(
    labeled_words: List[Tuple[Word, str, float]],
    pad_pre: float = 0.06,     # pre-roll padding in seconds
    pad_post: float = 0.06,    # post-roll padding in seconds
    min_span_ms: int = 80,     # drop extremely short edits
    merge_gap_ms: int = 150,   # join close spans of same label
) -> List[ActionSpan]:
    """
    Merge consecutive words with the same non-OK label into actionable spans.
    Padding is added to avoid cutting phonemes abruptly.
    """
    out: List[ActionSpan] = []

    # Helper to flush a current run
    def flush(cur_label: str, cur_words: List[Word], cur_ov: float):
        if not cur_words:
            return
        s = min(w.start for w in cur_words)
        e = max(w.end for w in cur_words)
        s = max(0.0, s - pad_pre)
        e = e + pad_post
        dur = e - s
        if dur * 1000.0 < min_span_ms:
            return
        confs = [w.confidence for w in cur_words if w.confidence is not None]
        avg_conf = float(np.mean(confs)) if confs else None
        min_conf = float(np.min(confs)) if confs else None
        # Overlap ratio: sum of per-word overlaps divided by span duration (clamped to [0,1])
        overlap_ratio = float(np.clip(cur_ov / max(1e-6, dur), 0.0, 1.0))
        out.append(ActionSpan(
            label=cur_label,
            start=s,
            end=e,
            words=cur_words.copy(),
            avg_conf=avg_conf,
            min_conf=min_conf,
            overlap_ratio=overlap_ratio
        ))

    # Walk through labeled words and group by label
    cur_label = None
    cur_words: List[Word] = []
    cur_overlap_sum = 0.0

    def same_label(a: Optional[str], b: Optional[str]) -> bool:
        return a == b

    for w, lab, ov in labeled_words:
        if lab == "OK":
            # Finish any active run
            flush(cur_label, cur_words, cur_overlap_sum)
            cur_label, cur_words, cur_overlap_sum = None, [], 0.0
            continue
        if same_label(cur_label, lab):
            cur_words.append(w)
            cur_overlap_sum += ov
        else:
            # New label run
            flush(cur_label, cur_words, cur_overlap_sum)
            cur_label = lab
            cur_words = [w]
            cur_overlap_sum = ov

    # Flush tail
    flush(cur_label, cur_words, cur_overlap_sum)

    # Merge spans of the same label that are very close to each other
    if not out:
        return out
    out.sort(key=lambda s: (s.start, s.end))
    merged: List[ActionSpan] = []
    gap_s = merge_gap_ms / 1000.0

    for span in out:
        if not merged:
            merged.append(span)
            continue
        last = merged[-1]
        if (span.label == last.label) and (span.start - last.end <= gap_s):
            # merge
            new_words = last.words + span.words
            s = last.start
            e = max(last.end, span.end)
            dur = e - s
            # recompute confidence stats & overlap ratio
            confs = [w.confidence for w in new_words if w.confidence is not None]
            avg_conf = float(np.mean(confs)) if confs else None
            min_conf = float(np.min(confs)) if confs else None
            # approximate overlap ratio by weighted average
            ov_ratio = (last.overlap_ratio * (last.end - last.start) +
                        span.overlap_ratio * (span.end - span.start)) / max(1e-6, dur)
            merged[-1] = ActionSpan(label=span.label, start=s, end=e, words=new_words,
                                    avg_conf=avg_conf, min_conf=min_conf, overlap_ratio=float(np.clip(ov_ratio, 0.0, 1.0)))
        else:
            merged.append(span)

    return merged


# =========================
# End-to-end function
# =========================

def analyze_recording_with_whisper_fusion(
    wav_path: str,
    whisper_json_or_path: Any,
    vad_mode: int = 2,          # 0 permissive (may let background in), 3 strict (only clean speech).
    # acoustic thresholds
    rms_db_boost: float = 6.0,  # how much louder than the background a frame must be to count as “loud.”
    flat_th: float = 0.35,      # higher → only very noisy textures are marked.
    flux_z: float = 1.5,        # controls sensitivity to sudden changes.
    zcr_th: float = 0.2,        # controls detection of crackles / high-frequency artifacts.
    # fusion thresholds
    low_conf_th: float = 0.58,
    min_overlap_for_effect: float = 0.02,
    # span shaping
    span_pad_pre: float = 0.06,
    span_pad_post: float = 0.06,
    span_min_ms: int = 80,
    span_merge_gap_ms: int = 150,
) -> Tuple[List[ProblemSegment], List[ActionSpan]]:
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

    # 3) Acoustic defect events restricted to speech
    events = detect_acoustic_events_in_speech(
        y, sr, vad_mask, hop_len,
        rms_db_boost=rms_db_boost,
        flat_th=flat_th,
        flux_z=flux_z,
        zcr_th=zcr_th,
        min_ms=30,             # keep short min here; we'll shape spans later
        merge_gap_ms=120
    )

    # 4) Load Whisper words
    if isinstance(whisper_json_or_path, str):
        with open(whisper_json_or_path, "r", encoding="utf-8") as f:
            whisper_json = json.load(f)
    else:
        whisper_json = whisper_json_or_path
    words = load_whisper_words(whisper_json)

    # 5) Per-word labeling based on fusion
    labeled = label_words_by_fusion(
        words,
        events,
        low_conf_th=low_conf_th,
        min_overlap_for_effect=min_overlap_for_effect
    )

    # 6) Convert to action spans (with padding, merging, stats)
    spans = spans_from_labeled_words(
        labeled,
        pad_pre=span_pad_pre,
        pad_post=span_pad_post,
        min_span_ms=span_min_ms,
        merge_gap_ms=span_merge_gap_ms
    )

    return events, spans