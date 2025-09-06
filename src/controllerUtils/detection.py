from typing import List, Optional, Tuple, Iterable, Set, Dict, Any
from src.stitcher.models.stitcherModels import WordToken
import json
import sys
from pathlib import Path
import logging
import os

# Make `src/` importable for the project layout
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.controllerUtils.energy_scorer import (
    detect_energy
)

from src.controllerUtils.plot import (
    plot_speech_spectrogram
)

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
    # Whisper transcription:
    whisper_json_or_path,
    low_conf_th: float = 0.58, # confidence threshold for low-confidence words
    # Context anomalies:
    anomaly_word_idx: Optional[Iterable[int]] = None,
    # Calculated fusion weights / threshold:
    w_conf_w: float = 0.70,      # weight for confidence term (low confidence => more likely to synth)
    energy_w: float = 0.90,      # weight for energy overlap term
    anomaly_w: float = 0.60,     # weight for anomaly term
    synth_score_th: float = 0.60, # threshold to decide synthesis
    # --- Gap handling params ---
    gap_min_dur: float = 0.12,          # ignore tiny gaps
    gap_energy_cov_th: float = 0.30,     # fraction of gap covered by energy to consider it noise
    gap_energy_score_th: float = 0.75,  # min energy score to consider gap as noise
    # --- Plotting params (if plotting is enabled) ---
    plot_spectrogram: bool = True,
):
    try:
        # --- 1) Run the acoustic analysis (tags masking vs non-masking) ---
        energy_events = detect_energy(wav_path)

        def energy_overlap(t0: float, t1: float) -> Tuple[float, float]:
            ov = 0.0
            score = 0.0
            for iv in energy_events:
                if iv['label'] != 'loud_noise':
                    continue
                temp_ov = _overlap_seconds((t0, t1), (iv['start_time'], iv['end_time']))
                if temp_ov > 0.0:
                    score = max(score, float(iv.get('score', 0.0)))
                    ov += temp_ov
            return ov, score

        # --- 2) Load Whisper words ---
        if isinstance(whisper_json_or_path, str):
            with open(whisper_json_or_path, "r", encoding="utf-8") as f:
                whisper_json = json.load(f)
        else:
            whisper_json = whisper_json_or_path
        words = load_whisper_words(whisper_json)

        # --- 3) Prepare anomaly index sets ---
        word_anom: Set[int] = set(int(i) for i in (anomaly_word_idx or []))
        word_anom = {i for i in word_anom if 0 <= i < len(words)}

        # --- 4) Min word length in transcription ---
        min_word_len = words[0].end - words[0].start if words else 0.2
        for w in words:
            wl = w.end - w.start
            min_word_len = min(min_word_len, wl)

        # --- 5) Score + decide per existing word + insert gap blanks ---
        tokens: List[WordToken] = []

        for i, w in enumerate(words):

            # Confidence flag (low confidence => True)
            conf_low = (w.confidence is not None) and (float(w.confidence) <= low_conf_th)

            # Use only masking events when evaluating words (noise overlapping speech)
            ov, energy_score = energy_overlap(w.start, w.end)
            overlap_norm = _clamp(ov / 0.15, 0.0, 1.0)

            in_anomaly = 1.0 if i in word_anom else 0.0
            conf_term = 1.0 if conf_low else 0.0

            score = (w_conf_w * (1.0 - conf_term)) + (energy_w * overlap_norm * energy_score) + (anomaly_w * in_anomaly)
            to_synth = (score >= synth_score_th)

            out_text = "[blank]" if (to_synth and (i in word_anom)) else w.text

            tokens.append(
                WordToken(
                    start=float(w.start),
                    end=float(w.end),
                    text=out_text,
                    to_synth=bool(to_synth and (i in word_anom)),
                    is_speech=True,
                    synth_path=None,
                )
            )

        # --- 6) Insert [blank] tokens for gaps likely to be noise ---
        for i in range(len(tokens) - 1):
            w0 = tokens[i]
            w1 = tokens[i + 1]
            gap_dur = w1.start - w0.end
            if gap_dur < gap_min_dur or gap_dur < min_word_len:
                continue

            # Gap is long enough to consider
            ov, energy_score = energy_overlap(w0.end, w1.start)
            gap_frac = ov / gap_dur if gap_dur > 0.0 else 0.0

            # Check if gap is mostly noise
            is_noise = (gap_frac >= gap_energy_cov_th and energy_score >= gap_energy_score_th)

            if is_noise and not w0.to_synth and not w1.to_synth:
                # Insert a blank token for the gap
                tokens.append(
                    WordToken(
                        start=float(w0.end),
                        end=float(w1.start),
                        text="[blank]",
                        to_synth=True,
                        is_speech=True,
                        synth_path=None,
                    )
                )

        tokens.sort(key=lambda t: (t.start, t.end))

        # --- 7) Plot spectrogram with overlays ---

        if plot_spectrogram:
            # prepare anomaly segments for plotting
            anom_segs = [words[i] for i in word_anom if 0 <= i < len(words)]

            plot_speech_spectrogram(
                wav_path = wav_path,
                out_path = "log/spectrogram.png",
                energy_segments = energy_events,
                anomaly_segments= anom_segs,
                low_conf_segments= words,
                conf_threshold = low_conf_th
            )

    except Exception as e:
        detection_logger.error(f"Error in build_word_tokens_of_detection: {e}")
        tokens = []

    return tokens
