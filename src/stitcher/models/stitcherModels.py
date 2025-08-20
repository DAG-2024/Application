from pydantic import BaseModel
from typing import List, Optional
from pydantic import TypeAdapter
import json

class WordToken(BaseModel):
    start: float
    end: float
    text: str
    to_synth: bool
    is_speech: bool
    synth_path: Optional[str] = None  # filled in after TTS

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


class FixResponse(BaseModel):
    fixed_url: str

    
