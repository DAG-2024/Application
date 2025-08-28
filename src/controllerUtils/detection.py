from typing import List, Optional, Tuple, Iterable, Callable, Set
from stitcher.models.stitcherModels import WordToken
from pydantic import BaseModel
import json
import numpy as np
import librosa

from .noise_event_detection import (
    analyze_recording_with_whisper_fusion,
    load_whisper_words,
    webrtc_speech_mask,
    evaluate_word_masking
)

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
    anomaly_gap_idx: Optional[Iterable[int]] = None,   # insert [blank] between k and k+1
    # Calculated fusion weights / threshold:
    w_conf_w: float = 0.55,        # weight for low-confidence term
    acoustic_w: float = 0.30,      # weight for acoustic-overlap term
    anomaly_w: float = 0.40,       # weight for anomaly flag
    synth_score_th: float = 0.60,  # final score threshold for resynthesis
    # Inserted [blank] shaping:
    insert_min_window: float = 0.06,  # if gap too small, create this duration
    insert_margin: float = 0.015,     # margin away from neighbors (s)
    # Masking knobs
    mask_score_th: float = 0.65,   # ≥ th => force synth + [blank]
    vad_non_speech_frac: float = 0.60,
    low_speech_band_frac: float = 0.35,
    high_flatness: float = 0.50,
    very_low_rms_db: float = -40.0,
    very_high_rms_db: float = -2.0,
    low_pitch_conf_frac: float = 0.70,
):
    """
    High-level helper that:
      1) Runs acoustic+confidence fusion (VAD + features + Whisper confidences).
      2) Integrates context anomalies in a *calculated* way.
      3) Replaces anomalous words with [blank] and inserts [blank] tokens in requested gaps.
      4) Returns List[WordToken] (in chronological order) and acoustic defect spans.

    anomaly_word_idx:
        Indices of words (0-based, order defined by parsed Whisper words) that are wrong and
        should be replaced with [blank].

    anomaly_gap_idx:
        Indices 'k' meaning "insert a [blank] between words[k] and words[k+1]".
        If k is -1, we treat it as insertion *before* the first word; if k == len(words)-1,
        insertion *after* the last word (edge cases handled).

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
    acoustic_events, action_spans = analyze_recording_with_whisper_fusion(
        wav_path=wav_path,
        whisper_json_or_path=whisper_json_or_path,
        vad_mode=vad_mode,
        low_conf_th=low_conf_th,
    )

    # --- 2) Load Whisper words in canonical order (defines indices) ---
    if isinstance(whisper_json_or_path, str):
        with open(whisper_json_or_path, "r", encoding="utf-8") as f:
            whisper_json = json.load(f)
    else:
        whisper_json = whisper_json_or_path
    words = load_whisper_words(whisper_json)   # Word(text,start,end,confidence)

    # --- 3) VAD speech intervals for is_speech decisions ---
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    vad_mask, _, hop_len = webrtc_speech_mask(y, sr, mode=vad_mode)
    hop_s = hop_len / sr
    speech_intervals = _mask_to_time_intervals(vad_mask, hop_s)

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

    if anomaly_gap_idx is not None:
        gap_anom: Set[int] = set(int(i) for i in (anomaly_gap_idx or []))
        # Clip indices to range
        # gaps can be -1 (before first) .. len(words)-1 (after last)
        gap_anom = {i for i in gap_anom if -1 <= i <= len(words) - 1}
    else:
        gap_anom: Set[int] = set()

    # --- 5) Word-level masking detection (catches fully masked words) ---
    mask_scores = evaluate_word_masking(
        y=y, sr=sr, words=words, vad_mask=vad_mask, hop_len=hop_len,
        vad_non_speech_frac=vad_non_speech_frac,
        low_speech_band_frac=low_speech_band_frac,
        high_flatness=high_flatness,
        very_low_rms_db=very_low_rms_db,
        very_high_rms_db=very_high_rms_db,
        low_pitch_conf_frac=low_pitch_conf_frac,
    )

    # --- 6) Score + decide per existing word ---
    tokens: List[WordToken] = []
    for i, w in enumerate(words):
        # Confidence normalization
        conf_raw = w.confidence if (w.confidence is not None) else 0.5
        conf_norm = _clamp(conf_raw, 0.0, 1.0)

        # Acoustic overlap normalized to ~0..1 using a 150ms scale
        ov = acoustic_overlap(w.start, w.end)
        overlap_norm = _clamp(ov / 0.15, 0.0, 1.0)


        masked = (mask_scores[i] >= mask_score_th)

        # Anomaly flag
        in_anomaly = 1.0 if i in word_anom else 0.0

        # Final score
        score = (w_conf_w * (1.0 - conf_norm)) + (acoustic_w * overlap_norm) + (anomaly_w * in_anomaly)

        to_synth = (score >= synth_score_th) or masked

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

    # --- 7) Insert [blank] tokens for requested gaps ---
    # We insert *after* building the base list so indices referenced by the caller
    # remain stable (we’ll sort by time afterwards).
    inserted: List[WordToken] = []

    def make_insert_token(left_end: float, right_start: float) -> Tuple[float, float]:
        """
        Decide the time window for an inserted token between (left_end, right_start).
        If natural gap is sufficient, use it with margins; else synthesize a small
        window centered between neighbors.
        """
        natural_gap = max(0.0, right_start - left_end)
        if natural_gap >= max(insert_min_window, 2 * insert_margin):
            s = left_end + insert_margin
            e = right_start - insert_margin
            return s, e
        # Create synthetic centered window
        mid = (left_end + right_start) / 2.0
        half = insert_min_window / 2.0
        s = mid - half
        e = mid + half
        # Clamp so we don't cross neighbors too much
        s = max(s, left_end + 1e-3)
        e = min(e, right_start - 1e-3) if right_start > left_end else s + insert_min_window
        if e <= s:
            e = s + insert_min_window
        return s, e

    for k in sorted(gap_anom):
        if len(words) == 0:
            # No words at all: place a token at t=0..insert_min_window
            s, e = 0.0, insert_min_window
        elif k == -1:
            # Insert before first word
            right_start = words[0].start
            left_end = max(0.0, right_start - insert_min_window)  # fabricate a left boundary
            s, e = make_insert_token(left_end, right_start)
        elif k == len(words) - 1:
            # Insert after last word
            left_end = words[-1].end
            right_start = left_end + insert_min_window  # fabricate a right boundary
            s, e = make_insert_token(left_end, right_start)
        else:
            # Insert between words[k] and words[k+1]
            left_end = words[k].end
            right_start = words[k + 1].start
            s, e = make_insert_token(left_end, right_start)

        inserted.append(
            WordToken(
                start=float(s),
                end=float(e),
                text="[blank]",
                to_synth=True,  # insertion must be synthesized
                is_speech=is_speech_interval(s, e),
                synth_path=None,
            )
        )

    # --- 8) Merge and sort all tokens by time ---
    all_tokens = tokens + inserted
    all_tokens.sort(key=lambda t: (t.start, t.end))

    # --- 9) Optional: return acoustic spans for denoisers ---
    noise_spans = [(ev.start, ev.end) for ev in acoustic_events]

    return all_tokens, noise_spans
