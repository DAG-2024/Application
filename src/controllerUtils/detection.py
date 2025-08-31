from typing import List, Optional, Tuple, Iterable, Set, Dict, Any

from scipy.fft import prev_fast_len
from stitcher.models.stitcherModels import WordToken
from pydantic import BaseModel
import json
import numpy as np
import librosa

from .noise_event_detection import (
    detect_noise_events,
    webrtc_speech_mask,
    mask_to_segments
)

import logging
import os

# Configure logging
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
detection_logger = logging.getLogger("detection")
detection_logger.setLevel(logging.DEBUG)

# Avoid adding multiple handlers if re-imported
if not detection_logger.handlers:
    file_handler = logging.FileHandler(os.path.join(log_dir, "detection.log"))
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    detection_logger.addHandler(file_handler)

# =========================
# Data structures
# =========================

from dataclasses import dataclass

@dataclass
class Word:
    """Minimal word representation from Whisper output."""
    text: str
    start: float
    end: float
    confidence: Optional[float]  # Some Whisper builds may not produce confidences

# ------------------------------------------------------------
# Adapter: produce your WordToken list from audio + whisper JSON
# ------------------------------------------------------------
def _overlap_seconds(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Duration of overlap (seconds) between intervals a=[a0,a1], b=[b0,b1]."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp value v to [lo, hi]."""
    return max(lo, min(hi, v))

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
        for w in seg["words"]:
            # Some flavors use "word" (string with leading space), others "text".
            text = w.get("word") or w.get("text") or ""
            start = float(w.get("start", seg.get("start", 0.0)))
            end = float(w.get("end", seg.get("end", start)))
            conf = w.get("score", None)
            if conf is not None:
                conf = float(conf)
            words.append(Word(text=text.strip(), start=start, end=end, confidence=conf))
    return words


# ------------------------------------------------------------
# Adapter: context anomalies + [blank] replacement/insertion
# ------------------------------------------------------------
def build_word_tokens_of_detection(
    wav_path: str,
    whisper_json_or_path,
    # VAD + fusion tuning
    low_conf_th: float = 0.58,
    vad_mode: int = 3,
    # Context anomalies:
    anomaly_word_idx: Optional[Iterable[int]] = None,
    # Calculated fusion weights / threshold:
    w_conf_w: float = 0.70,
    acoustic_w: float = 0.30,
    anomaly_w: float = 0.70,
    synth_score_th: float = 0.60,
    # Frame indices for VAD + acoustic analysis
    frame_ms: int = 10,
    hop_ms: int = 5,
    # Acoustic insertion parameters
    rms_db_boost: float = 12.0, # dB above noise floor
    flat_th: float = 0.8,   # spectral flatness threshold
    flux_z: float = 5,      # spectral flux z-score threshold
    zcr_th: float = 0.5,    # zero-crossing rate threshold
    # Masking classification threshold (fraction of speech-overlap inside an event)
    speech_overlap_th: float = 0.30,
    # --- Gap handling params ---
    gap_min_dur: float = 0.12,          # ignore tiny gaps
    gap_noise_cov_th: float = 0.70,     # fraction of gap covered by non-masking noise
    gap_event_score_th: float = 0.60,   # min event score to consider for noise-driven blanks
    gap_from_speech_min_overlap: float = 0.60,  # fraction of gap overlapped by VAD speech
    handle_leading_trailing_gaps: bool = True,
):
    # --- 1) Run the acoustic analysis (tags masking vs non-masking) ---
    acoustic_events = detect_noise_events(
        wav_path=wav_path,
        hop_ms=hop_ms,
        vad_mode=vad_mode,
        rms_db_boost=rms_db_boost,
        flat_th=flat_th,
        flux_z=flux_z,
        zcr_th=zcr_th,
        min_event_dur=0.05,
        min_gap=0.05,
        speech_overlap_th=speech_overlap_th,
    )

    detection_logger.debug(f"Computed {len(acoustic_events)} defect intervals from acoustic analysis.")
    detection_logger.debug(f"Defect intervals: {acoustic_events}")

    masking_events = [e for e in acoustic_events if e.get('masking_speech', False)]
    non_masking_events = [e for e in acoustic_events if not e.get('masking_speech', False)]
    detection_logger.debug(f"Masking events: {len(masking_events)}, Non-masking events: {len(non_masking_events)}")

    def acoustic_overlap(t0: float, t1: float, only_masking: bool = True) -> Tuple[float, float]:
        evs = masking_events if only_masking else acoustic_events
        ov = 0.0
        score = 0.0
        for iv in evs:
            temp_ov = _overlap_seconds((t0, t1), (iv['start'], iv['end']))
            if temp_ov > 0.0:
                score = max(score, float(iv.get('score', 0.0)))
                ov += temp_ov
        return ov, score

    def acoustic_overlap_in(evs: List[Dict[str, Any]], t0: float, t1: float) -> Tuple[float, float]:
        ov = 0.0
        score = 0.0
        for iv in evs:
            temp_ov = _overlap_seconds((t0, t1), (iv['start'], iv['end']))
            if temp_ov > 0.0:
                score = max(score, float(iv.get('score', 0.0)))
                ov += temp_ov
        return ov, score

    # --- 3) Load Whisper words ---
    if isinstance(whisper_json_or_path, str):
        with open(whisper_json_or_path, "r", encoding="utf-8") as f:
            whisper_json = json.load(f)
    else:
        whisper_json = whisper_json_or_path
    words = load_whisper_words(whisper_json)

    detection_logger.debug(f"Loaded {len(words)} words from Whisper JSON.")

    # --- 4) VAD speech intervals ---
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    audio_dur = float(len(y) / sr)
    vad_mask, _, hop_len = webrtc_speech_mask(y, sr, mode=vad_mode, frame_ms=frame_ms, hop_ms=hop_ms)
    hop_s = hop_len / sr
    speech_intervals = mask_to_segments(vad_mask, hop_s)

    detection_logger.debug(f"Computed {len(speech_intervals)} speech intervals from VAD.")
    detection_logger.debug(f"Speech intervals: {speech_intervals}")

    def is_speech_interval(t0: float, t1: float, min_overlap: float = 0.2, percentage: bool = False) -> bool:
        for iv in speech_intervals:
            ov = _overlap_seconds((t0, t1), iv)
            if percentage and ov / max(1e-9, (t1 - t0)) >= min_overlap:
                return True
            if not percentage and ov >= min_overlap:
                return True
        return False

    def speech_overlap_ratio(t0: float, t1: float) -> float:
        total = max(1e-9, (t1 - t0))
        ov = 0.0
        for iv in speech_intervals:
            ov += _overlap_seconds((t0, t1), iv)
        return _clamp(ov / total, 0.0, 1.0)

    def intersect_intervals(base: Tuple[float, float], segs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        b0, b1 = base
        out: List[Tuple[float, float]] = []
        for s, e in segs:
            s2, e2 = max(b0, s), min(b1, e)
            if e2 > s2:
                out.append((s2, e2))
        return out

    def merge_intervals(segs: List[Tuple[float, float]], min_dur: float = 0.0, min_gap: float = 0.03) -> List[Tuple[float, float]]:
        if not segs:
            return []
        segs = sorted(segs)
        out: List[Tuple[float, float]] = []
        cs, ce = segs[0]
        for s, e in segs[1:]:
            if s - ce <= min_gap:
                ce = max(ce, e)
            else:
                if (ce - cs) >= min_dur:
                    out.append((cs, ce))
                cs, ce = s, e
        if (ce - cs) >= min_dur:
            out.append((cs, ce))
        return out

    def build_gap_blanks(g0: float, g1: float) -> List[WordToken]:
        # Skip tiny gaps fast
        if (g1 - g0) < gap_min_dur:
            return []

        # Coverage by non-masking noise
        nm_ov, nm_score_max = acoustic_overlap_in(non_masking_events, g0, g1)
        nm_cov = nm_ov / max(1e-9, (g1 - g0))
        loud_gap = (nm_cov >= gap_noise_cov_th) and (nm_score_max >= gap_event_score_th)

        # Coverage by VAD speech
        sp_cov = speech_overlap_ratio(g0, g1)
        speech_gap = (sp_cov >= gap_from_speech_min_overlap)

        if not (loud_gap or speech_gap):
            return []

        # Build blank spans clipped to speech +/- high-scoring non-masking noise inside gap
        blank_spans: List[Tuple[float, float]] = []
        # Speech-driven spans
        blank_spans += intersect_intervals((g0, g1), speech_intervals)
        # Noise-driven spans (only events above score threshold)
        high_nm = [(e['start'], e['end']) for e in non_masking_events if float(e.get('score', 0.0)) >= gap_event_score_th]
        blank_spans += intersect_intervals((g0, g1), high_nm)

        # Merge and filter
        blank_spans = merge_intervals(blank_spans, min_dur=gap_min_dur, min_gap=0.03)

        # Emit WordTokens
        tokens_out: List[WordToken] = []
        for s, e in blank_spans:
            # is_speech if this span has material VAD speech
            is_sp = is_speech_interval(s, e, min_overlap=0.2, percentage=False) or (speech_overlap_ratio(s, e) >= 0.5)
            tokens_out.append(
                WordToken(
                    start=float(s),
                    end=float(e),
                    text="[blank]",
                    to_synth=True,
                    is_speech=True,
                    synth_path=None,
                )
            )
        return tokens_out

    # --- 5) Prepare anomaly index sets ---
    word_anom: Set[int] = set(int(i) for i in (anomaly_word_idx or []))
    word_anom = {i for i in word_anom if 0 <= i < len(words)}

    # --- 6) Min word length in transcription ---
    min_word_len = words[0].end - words[0].start if words else 0.2
    for w in words:
        wl = w.end - w.start
        min_word_len = min(min_word_len, wl)
    detection_logger.debug(f"Min word length: {min_word_len}")

    # --- 7) Score + decide per existing word + insert gap blanks ---
    tokens: List[WordToken] = []
    prev_end = 0.0

    # Leading gap
    if handle_leading_trailing_gaps and words:
        tokens.extend(build_gap_blanks(0.0, max(0.0, words[0].start)))

    for i, w in enumerate(words):
        # Insert blanks for gap before this word
        if i > 0 and handle_leading_trailing_gaps:
            tokens.extend(build_gap_blanks(prev_end, w.start))

        # Confidence flag (low confidence => True)
        conf_low = (w.confidence is not None) and (float(w.confidence) <= low_conf_th)

        # Use only masking events when evaluating words (noise overlapping speech)
        ov, acoustic_score = acoustic_overlap(w.start, w.end, only_masking=True)
        overlap_norm = _clamp(ov / 0.15, 0.0, 1.0)

        in_anomaly = 1.0 if i in word_anom else 0.0
        conf_term = 1.0 if conf_low else 0.0

        score = (w_conf_w * (1.0 - conf_term)) + (acoustic_w * overlap_norm * acoustic_score) + (anomaly_w * in_anomaly)
        to_synth = (score >= synth_score_th)

        out_text = "[blank]" if (to_synth and (i in word_anom)) else w.text

        tokens.append(
            WordToken(
                start=float(w.start),
                end=float(w.end),
                text=out_text,
                to_synth=bool(to_synth),
                is_speech=is_speech_interval(w.start, w.end),
                synth_path=None,
            )
        )

        prev_end = w.end

    # Trailing gap
    if handle_leading_trailing_gaps and words and audio_dur > prev_end:
        tokens.extend(build_gap_blanks(prev_end, audio_dur))

    tokens.sort(key=lambda t: (t.start, t.end))
    return tokens