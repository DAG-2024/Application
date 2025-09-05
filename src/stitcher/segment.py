# segments.py
"""
Builds a stitched-timeline 'segments' list for the UI WITHOUT touching your
existing stitching algorithm. It *simulates* clip boundaries exactly like
your stitch_all() does, then merges adjacent clips of the same source into
coarser 'segments' that are easy to color on a waveform.
"""
from typing import List, Dict, Optional
import os
import ffmpeg
from .models.stitcherModels import WordToken, Segment

# must match your acrossfade in stitch_all()
DEFAULT_CROSSFADE = 0.02  # seconds


def _ffprobe_duration(path: str) -> float:
    """
    Get clip duration (seconds) using ffprobe (via ffmpeg-python).
    Used to time synthesized clips.
    """
    info = ffmpeg.probe(path)
    return float(info["format"]["duration"])


def _simulate_token_clip_duration(
    token: WordToken, prev_end: float
) -> Optional[float]:
    """
    Simulate the *per-token* clip duration used by your stitch_all():
      - if token is synth: duration = duration of token.synth_path
      - if token is orig and is_speech: duration = token.end - prev_end
      - if token is not active speech: None (skipped)
    Returns None if the token isn't included as a clip.
    """
    if not token.is_speech:
        return None

    if token.to_synth and token.synth_path:
        return _ffprobe_duration(token.synth_path)

    if not token.to_synth:
        return max(0.0, token.end - prev_end)

    return None


def build_segments_for_ui(
    tokens: List[WordToken],
    crossfade: float = DEFAULT_CROSSFADE,
) -> List[Segment]:
    """
    Construct a compact list of UI segments that matches your stitching timeline:
    1) Simulate the exact per-token clip durations (like stitch_all()).
    2) Merge adjacent tokens with the same source ('orig' or 'synth') into segments.
    3) Account for crossfades between *every* adjacent clip (internal + between segments).

    Returns a list[Segment] ready to be serialized in FixResponse.
    """
    # Step 1: compute per-token clip durations with the same prev_end logic
    per_token_dur: List[Optional[float]] = [None] * len(tokens)
    prev_end = 0.0
    for i, t in enumerate(tokens):
        dur = _simulate_token_clip_duration(t, prev_end)
        per_token_dur[i] = dur
        # stitch_all() advances prev_end after EVERY token (speech or synth)
        # because it sets prev_end = s.end + 0.001; for synth tokens we don't
        # know end in original, so we still advance using the original end to
        # keep alignment with "orig" trimming that follows.
        if t.is_speech:
            prev_end = t.end + 0.001

    # Step 2: scan and merge into runs by source
    runs: List[Dict] = []
    cur: Dict = {}

    def src_of(tok: WordToken) -> str:
        return "synth" if (tok.to_synth and tok.synth_path) else "orig"

    for idx, (tok, dur) in enumerate(zip(tokens, per_token_dur)):
        if dur is None:
            continue  # not a clip in stitch_all()
        src = src_of(tok)
        if cur and cur["source"] == src:
            cur["tokens"].append(idx)
            cur["texts"].append(tok.text)
            cur["clip_durations"].append(dur)
            if src == "orig":
                cur["src_end"] = tok.end
        else:
            if cur:
                runs.append(cur)
            cur = {
                "source": src,
                "tokens": [idx],
                "texts": [tok.text],
                "clip_durations": [dur],
                "src_start": tok.start if src == "orig" else None,
                "src_end": tok.end if src == "orig" else None,
            }
    if cur:
        runs.append(cur)

    # Step 3: compute stitched timeline with crossfades at every clip boundary
    # For a run with K clips, internal overlaps subtract (K-1)*crossfade.
    # Between runs, there is an additional single crossfade shared by both sides.
    segments: List[Segment] = []
    t = 0.0
    for i, r in enumerate(runs):
        k = len(r["clip_durations"])
        raw_sum = sum(r["clip_durations"])
        internal_overlap = max(0, k - 1) * crossfade
        seg_duration = max(0.0, raw_sum - internal_overlap)

        seg_start = t
        seg_end = seg_start + seg_duration

        overlap_out = crossfade if i < len(runs) - 1 else 0.0
        overlap_in = crossfade if i > 0 else 0.0

        segments.append(Segment(
            start=seg_start,
            end=seg_end,
            source=r["source"],
            tokens=r["tokens"],
            text=" ".join(r["texts"]),
            src_start=r.get("src_start"),
            src_end=r.get("src_end"),
            overlap_in=overlap_in,
            overlap_out=overlap_out,
        ))

        # next segment starts before this ends by the crossfade amount
        t = seg_end - overlap_out

    return segments
