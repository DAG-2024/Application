from pydantic import BaseModel
from typing import List, Optional



class WordToken(BaseModel):
    start: float
    end: float
    text: str
    to_synth: bool
    is_speech: bool
    synth_path: Optional[str] = None  # filled in after TTS


class FixResponse(BaseModel):
    fixed_url: str

    