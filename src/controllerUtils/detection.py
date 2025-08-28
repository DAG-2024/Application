from typing import List, Optional, Tuple, Iterable, Callable, Set
from stitcher.models.stitcherModels import WordToken
from pydantic import BaseModel
import json
import numpy as np
import librosa

from .noise_event_detection import (
    analyze_recording,
    webrtc_speech_mask,
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


# ------------------------------------------------------------
# Adapter: produce your WordToken list from audio + whisper JSON
# ------------------------------------------------------------
def _mask_to_time_intervals(mask: np.ndarray, hop_s: float) -> List[Tuple[float, float]]:
    """Convert boolean frame mask to [start,end] intervals (seconds)."""
    intervals: List[Tuple[float, float]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            intervals.append((i * hop_s, j * hop_s))
            i = j
        else:
            i += 1
    return intervals

def _overlap_seconds(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Duration of overlap (seconds) between intervals a=[a0,a1], b=[b0,b1]."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

def _clamp(v: float, lo: float, hi: float) -> float:
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


# ------------------------------------------------------------
# Adapter: context anomalies + [blank] replacement/insertion
# ------------------------------------------------------------

def build_word_tokens_of_detecation(
    wav_path: str,
    whisper_json_or_path,
    # VAD + fusion tuning
    low_conf_th: float = 0.58,
    vad_mode: int = 2,
    # Context anomalies:
    anomaly_word_idx: Optional[Iterable[int]] = None,  # words to replace with [blank]
    # Calculated fusion weights / threshold:
    w_conf_w: float = 0.55,        # weight for low-confidence term
    acoustic_w: float = 0.30,      # weight for acoustic-overlap term
    anomaly_w: float = 0.40,       # weight for anomaly flag
    synth_score_th: float = 0.60,  # final score threshold for resynthesis
):
    """
    High-level helper that:
      1) Runs acoustic+confidence fusion (VAD + features + Whisper confidences).
      2) Integrates context anomalies in a *calculated* way.
      3) Replaces anomalous words with [blank] and inserts [blank] tokens in requested gaps.
      4) Returns List[WordToken] (in chronological order) and acoustic defect spans.

    anomaly_word_idx:
        Indices of words (0-based, order defined by parsed Whisper words) that are wrong and
        should be replaced with [blank]

    Decision score (per word):
        score = w_conf_w * (1 - conf_norm) + acoustic_w * overlap_norm + anomaly_w * is_in_anomaly
        where:
          conf_norm   = clamp(conf,0,1) if available else 0.5
          overlap_norm= clamp(overlap / 0.15, 0, 1)   # normalize 0..0.15s to [0..1]
          is_in_anomaly = 1 if word index in anomaly_word_idx else 0
        If score >= synth_score_th => to_synth=True.

    Inserted [blank] tokens:
        - Placed between adjacent words using their end/start times.
        - If the natural gap is < insert_min_window, we synthesize a small window
          centered between them, trimmed by insert_margin from neighbors.
    """
    # --- 1) Run the fused acoustic analysis ---
    acoustic_events, vad_events = analyze_recording(
        wav_path=wav_path,
        whisper_json_or_path=whisper_json_or_path,
        vad_mode=vad_mode,
        low_conf_th=low_conf_th,
    )

    detection_logger.debug(f"Acoustic events: {acoustic_events}")

    # --- 2) Load Whisper words in canonical order (defines indices) ---
    if isinstance(whisper_json_or_path, str):
        with open(whisper_json_or_path, "r", encoding="utf-8") as f:
            whisper_json = json.load(f)
    else:
        whisper_json = whisper_json_or_path
    words = load_whisper_words(whisper_json)   # Word(text,start,end,confidence)

    detection_logger.debug(f"Loaded {len(words)} words from Whisper JSON.")

    # --- 3) VAD speech intervals for is_speech decisions ---
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    vad_mask, _, hop_len = webrtc_speech_mask(y, sr, mode=vad_mode)
    hop_s = hop_len / sr
    speech_intervals = _mask_to_time_intervals(vad_mask, hop_s)

    detection_logger.debug(f"Computed {len(speech_intervals)} speech intervals from VAD.")
    detection_logger.debug(f"Speech intervals: {speech_intervals}")

    def is_speech_interval(t0: float, t1: float) -> bool:
        """Consider speech if ≥20 ms overlaps VAD=Speech."""
        min_overlap = 0.02
        for iv in speech_intervals:
            if _overlap_seconds((t0, t1), iv) >= min_overlap:
                return True
        return False

    # --- 4) Quick lookup for acoustic overlap per word (from action spans) ---
    # We compute the total overlap with *acoustic* spans (where defects live).
    # (Not using the fused label directly—we want overlap seconds for scoring.)
    defect_intervals = [(s.start, s.end) for s in acoustic_events]

    def acoustic_overlap(t0: float, t1: float) -> float:
        ov = 0.0
        for iv in defect_intervals:
            ov += _overlap_seconds((t0, t1), iv)
        return ov

    # --- 5) Prepare anomaly index sets ---
    word_anom: Set[int] = set(int(i) for i in (anomaly_word_idx or []))
    # Clip indices to range
    word_anom = {i for i in word_anom if 0 <= i < len(words)}

    # --- 6) Score + decide per existing word ---
    tokens: List[WordToken] = []
    for i, w in enumerate(words):
        # Confidence normalization
        conf_raw = w.confidence if (w.confidence is not None) else 0.5
        conf_norm = _clamp(conf_raw, 0.0, 1.0)

        # Acoustic overlap normalized to ~0..1 using a 150ms scale
        ov = acoustic_overlap(w.start, w.end)
        overlap_norm = _clamp(ov / 0.15, 0.0, 1.0)

        # Anomaly flag
        in_anomaly = 1.0 if i in word_anom else 0.0

        # Final score
        score = (w_conf_w * (1.0 - conf_norm)) + (acoustic_w * overlap_norm) + (anomaly_w * in_anomaly)

        to_synth = (score >= synth_score_th)

        # If this word is flagged as context-anomalous AND we’re synthesizing,
        # we *replace its text* with "[blank]" to be filled later by GPT->TTS.
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

    # --- 7) Merge and sort all tokens by time ---
    tokens.sort(key=lambda t: (t.start, t.end))

    # --- 8) Optional: return acoustic spans for denoisers ---
    noise_spans = [(ev.start, ev.end) for ev in acoustic_events]

    return tokens, noise_spans
