from pydantic import BaseModel, TypeAdapter, Field
from typing import List, Optional, Literal
from pathlib import Path
import json

class WordToken(BaseModel):
    start: float
    end: float
    text: str
    to_synth: bool
    is_speech: bool
    synth_path: Optional[str] = None  # filled in after TTS

class Segment(BaseModel):
    """
    A merged, human-friendly indicator line for the UI on the *stitched* timeline.
    - start/end: final stitched times
    - source:    "orig" (from original) or "synth" (from TTS)
    - tokens:    indices of WordToken objects included in this segment
    - text:      joined text (post-bleed)
    - src_start/src_end: (for orig) where it came from in the original file
    - overlap_in/out: crossfade overlap into/out of this segment (seconds)
    """
    start: float
    end: float
    source: Literal["orig", "synth"]
    tokens: List[int]
    text: str
    src_start: Optional[float] = None
    src_end: Optional[float] = None
    overlap_in: float = 0.0
    overlap_out: float = 0.0

def wordtokens_to_json(wordtokens: List[WordToken]) -> str:
    """
    Serialize a list of WordToken objects to a JSON string.
    """
    return json.dumps([w.model_dump() for w in wordtokens])

def wordtokens_from_json(json_str: str) -> List[WordToken]:
    """
    Deserialize a JSON string into a list of WordToken objects.
    """
    data = json.loads(json_str)
    return TypeAdapter(List[WordToken]).validate_python(data)


def save_wordtokens_json(wordtokens: List[WordToken], file_path: str, *, pretty: bool = False) -> None:
    """
    Save a list of WordToken objects to a JSON file at 'file_path'.
    If pretty=True, writes indented, UTF-8 JSON; otherwise uses wordtokens_to_json().
    """
    # Build JSON text
    if pretty:
        import json
        text = json.dumps([w.model_dump() for w in wordtokens], ensure_ascii=False, indent=2)
    else:
        text = wordtokens_to_json(wordtokens)

    # Ensure directory exists and write file
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def load_wordtokens_json(file_path: str) -> List[WordToken]:
    """
    Load a list of WordToken objects from a JSON file at 'file_path'.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    json_str = path.read_text(encoding="utf-8")
    return wordtokens_from_json(json_str)


class FixResponse(BaseModel):
    """
        Response payload for /fix-audio.
          fixed_url: HTTP-served URL of the final stitched WAV.
          segments:  UI timeline describing orig vs synth stretches.
        """
    fixed_url: str
    segments: List[Segment]

