from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

def word_predictor(transcription: str):
    # Calculate the absolute path to the root .env file
    current_file = Path(__file__)
    root_dir = current_file.parent.parent.parent  # Go up from src/controllerUtils/ to root
    env_path = root_dir / ".env"
    
    load_dotenv(dotenv_path=str(env_path))

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

    # The system prompt to set context
    system_prompt = ("""You are an expert language model tasked with restoring missing words in a transcription.
                          The transcript comes from a spoken lecture and contains several missing words, marked by the placeholder `[blank]`.

                          These blanks were inserted where either:
                          - The original audio was masked by noise, resulting in a completely missing word, or
                          - The audio was transcribed incorrectly and then replaced by a blank.

                          Your task is to use the **context of the entire transcription** and any **domain-specific clues** 
                          (such as technical language or topic relevance) to accurately predict the most plausible word for each `[blank]`.

                          Important guidelines:
                          - Predict exactly **one word** per `[blank]` placeholder.
                          - However, you may predict **two words** if the **first word is an indefinite or definite article**, such as `"a"`, `"an"`, or `"the"` (e.g., `"a cat"`, `"the tree"`).
                          - Ignore any punctuation that appears near the `[blank]`. It may be incorrect due to audio corruption. Rely on sentence meaning and grammar to make your decision.
                          - Preserve the rest of the transcription exactly as it is, replacing only the `[blank]` entries.
                          - Your output should be a single string — the full transcription with predicted words inserted in place of the `[blank]` placeholders.
                          - use the entire transcription as a strong reference and the transcription's subject to determine the missing word.
                          - Do not add any additional commentary or explanation; only return the completed transcription.
                          ---

                          ### Example (for format and reasoning only):

                          **Input:**  
                          "The tree is. [blank] Due to its height, it gets a lot of sunlight."

                          **Output:**  
                          "The tree is tall. Due to its height, it gets a lot of sunlight."

                          ---

                          Now complete the transcription by filling in the blanks. 
                       """
                     )

    # IMPORTANT: `model` is your DEPLOYMENT NAME for gpt-35-turbo in Azure
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The transcription:\n\n{transcription}"}
        ],
        temperature=0.2
    )
    # If you want the full SDK response object:
    # return response

    # If you just want the completed transcript string:
    return response.choices[0].message.content


import re
from typing import List, Tuple, Callable, Optional
from stitcher.models.stitcherModels import WordToken, wordtokens_to_json

# -------------------------------
# 1) Build the predictor input
# -------------------------------
def tokens_to_template(tokens: List["WordToken"]) -> Tuple[str, List[int]]:
    """
    Create a single string of the whole transcript with [blank] where needed.
    Returns:
        template_str: e.g., "hello [blank] world ."
        blank_indices: indices in `tokens` that are [blank] (in left-to-right order).
    Notes:
        - We only trust the predictor to fill [blank] slots. We preserve non-blank words as anchors.
        - We keep punctuation with minimal spacing normalization.
    """
    parts = []
    blank_indices = []
    for i, t in enumerate(tokens):
        if t.text.strip().lower() == "[blank]":
            parts.append("[blank]")
            blank_indices.append(i)
        else:
            parts.append(t.text)

    # Normalize spaces around punctuation for a cleaner prompt
    template = " ".join(parts)
    template = normalize_spaces(punct_space(template))
    return template, blank_indices


# -------------------------------
# 2) Apply prediction back to tokens
# -------------------------------
def apply_prediction_to_tokens(
    original_tokens: List["WordToken"],
    fixed_sentence: str,
    blank_indices: List[int],
    split_multiword: bool = True,
    max_split: int = 1,
) -> List["WordToken"]:
    """
    Map model's corrected sentence back into the [blank] slots.

    Strategy:
      - Use non-blank tokens as *anchors* in order.
      - Locate those anchors inside the fixed sentence (after normalization).
      - The text *between* consecutive anchors is the fill for the [blank] between them.

    If split_multiword=True and a fill contains multiple words, split it into up to `max_split`
    new tokens by dividing the original [blank] duration evenly.

    If aligning fails for a given region, we leave that [blank] unchanged (still "[blank]").
    """
    tokens = original_tokens[:]  # shallow copy (we will replace elements)

    # Prepare normalized versions for robust matching
    fixed_norm = normalize_spaces(punct_space(fixed_sentence))
    fixed_words = tokenize_preserve_punct(fixed_norm)

    # Build the *anchor sequence* from non-blanks (normalized)
    anchors = []
    for i, t in enumerate(tokens):
        if t.text.strip().lower() != "[blank]":
            anchors.append((i, t.text))

    # Edge case: no anchors at all -> assign the entire fixed sentence to the first blank (if exists)
    if not anchors and blank_indices:
        fill_text = fixed_norm.strip()
        i_blank = blank_indices[0]
        tokens = _replace_blank_with_fill(tokens, i_blank, fill_text, split_multiword, max_split)
        return tokens

    # Convert anchors to a normalized word list to search inside `fixed_words`
    anchor_word_seqs = [tokenize_preserve_punct(normalize_spaces(punct_space(txt))) for _, txt in anchors]

    # Locate each anchor sequence in fixed_words (greedy, left-to-right)
    # We keep the span (start_idx_in_fixed_words, end_idx_exclusive)
    anchor_positions: List[Tuple[int, int]] = []
    cursor = 0
    for aw in anchor_word_seqs:
        pos = find_subsequence(fixed_words, aw, start=cursor)
        if pos is None:
            # If we fail to find this anchor, abort precise alignment for this region.
            # We'll still try to fill using neighboring anchors where possible.
            anchor_positions.append((-1, -1))
        else:
            s, e = pos
            anchor_positions.append((s, e))
            cursor = e  # move past this anchor for the next search

    # Now, we walk through anchors and blanks in order and extract fills.
    # We need to know which blank sits between which two anchors.
    # Build a list of "segments": [ START, A0, B0(blank), A1, B1, A2, ... , END ]
    # Where Ax are anchors and Bx are blank indices between them.
    # Compute the linear order of tokens and interleave blanks.
    # We'll iterate token-by-token to identify blanks between anchors.
    nonblank_ptr = 0  # index into anchors
    prev_fixed_end = 0
    last_anchor_fixed_span = None

    # Helper: map token index -> anchor order index (or None)
    token_to_anchor_order = {idx: k for k, (idx, _) in enumerate(anchors)}

    # Keep a list of (blank_index, fixed_start, fixed_end) to fill later
    blanks_to_fill: List[Tuple[int, int, int]] = []

    # We pass through all tokens; when we hit a [blank], we’ll decide its fixed_sentence span
    # by using the nearest anchor to the left and the next anchor to the right.
    for ti, tok in enumerate(tokens):
        if tok.text.strip().lower() == "[blank]":
            left_anchor_order = None
            right_anchor_order = None

            # find left anchor order
            j = ti - 1
            while j >= 0:
                if j in token_to_anchor_order:
                    left_anchor_order = token_to_anchor_order[j]
                    break
                j -= 1

            # find right anchor order
            j = ti + 1
            while j < len(tokens):
                if j in token_to_anchor_order:
                    right_anchor_order = token_to_anchor_order[j]
                    break
                j += 1

            # Translate anchor orders to fixed-word spans
            if left_anchor_order is not None and anchor_positions[left_anchor_order] != (-1, -1):
                left_fixed_end = anchor_positions[left_anchor_order][1]
            else:
                left_fixed_end = 0  # beginning of sentence

            if right_anchor_order is not None and anchor_positions[right_anchor_order] != (-1, -1):
                right_fixed_start = anchor_positions[right_anchor_order][0]
            else:
                right_fixed_start = len(fixed_words)  # end of sentence

            # Define the candidate fill region for this blank
            s_fw = max(0, left_fixed_end)
            e_fw = max(s_fw, right_fixed_start)
            blanks_to_fill.append((ti, s_fw, e_fw))

    # Perform the fills
    for ti, s_fw, e_fw in blanks_to_fill:
        fill_words = fixed_words[s_fw:e_fw]
        fill_text = detokenize(fill_words).strip()
        if fill_text:
            tokens = _replace_blank_with_fill(tokens, ti, fill_text, split_multiword, max_split)
        # else: leave as "[blank]" (no usable fill found)

    # Sort by time again (splitting insertions may create tiny ordering changes)
    tokens.sort(key=lambda t: (t.start, t.end))
    return tokens


# -------------------------------
# Helpers: normalization & search
# -------------------------------
def punct_space(s: str) -> str:
    """Ensure spaces around basic punctuation to make tokenization stable."""
    # Add space around these punctuations
    s = re.sub(r'([.,!?;:])', r' \1 ', s)
    # Collapse extra spaces
    return re.sub(r'\s+', ' ', s).strip()

def normalize_spaces(s: str) -> str:
    """Lowercase + collapse spaces for matching; keep case for final text separately when needed."""
    return re.sub(r'\s+', ' ', s).strip()

def tokenize_preserve_punct(s: str) -> List[str]:
    """
    Tokenize into words + punctuation tokens, preserving punctuation as separate tokens.
    """
    return re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE)

def detokenize(tokens: List[str]) -> str:
    """
    Join tokens back into a normal string, fixing spaces before punctuation.
    """
    if not tokens:
        return ""
    s = " ".join(tokens)
    # Remove space before punctuation like "word ,"
    s = re.sub(r'\s+([.,!?;:])', r'\1', s)
    # Fix parentheses/brackets spacing lightly
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    return s.strip()

def find_subsequence(hay: List[str], needle: List[str], start: int = 0) -> Optional[Tuple[int, int]]:
    """
    Find a list `needle` inside list `hay` starting at position `start`.
    Returns (start_idx, end_idx_exclusive) or None.
    Very simple greedy match; robust enough for normalized text.
    """
    if not needle:
        return (start, start)
    n, m = len(hay), len(needle)
    for i in range(start, n - m + 1):
        if hay[i:i+m] == needle:
            return (i, i + m)
    return None


# ---------------------------------------
# Helpers: replacing or splitting blanks
# ---------------------------------------
def _replace_blank_with_fill(
    tokens: List["WordToken"],
    blank_idx: int,
    fill_text: str,
    split_multiword: bool,
    max_split: int = 1,
) -> List["WordToken"]:
    """
    Replace a single [blank] token at `blank_idx` with `fill_text`.
    If split_multiword=True and `fill_text` has multiple words, split the time evenly.
    """
    blank = tokens[blank_idx]
    # Simple heuristic for "words": split on whitespace; keep punctuation attached to neighbors
    fill_words = fill_text.split()
    if (not split_multiword) or len(fill_words) <= 1:
        # Single token replacement
        tokens[blank_idx].text = fill_text
        tokens[blank_idx].to_synth = True
        # keep original timings; stitching/TTs will stretch/fit as needed
        return tokens

    # Multiword: cap the split to avoid exploding tokens
    fill_words = fill_words[:max_split]
    n = len(fill_words)
    dur = max(1e-6, blank.end - blank.start)
    step = dur / n

    # Build replacement tokens
    new_tokens: List["WordToken"] = []
    cur = blank.start
    for w in fill_words:
        new_tokens.append(
            type(blank)(  # preserve pydantic model type
                start=cur,
                end=min(blank.end, cur + step),
                text=w,
                to_synth=True,
                is_speech=blank.is_speech,
                synth_path=None
            )
        )
        cur += step

    # Replace in the list
    tokens = tokens[:blank_idx] + new_tokens + tokens[blank_idx+1:]
    return tokens


# ---------------------------------------
# 3) High-level convenience wrapper
# ---------------------------------------
def predict_and_fill_tokens(
    tokens: List["WordToken"],
    predictor: Callable[[str], str],
    split_multiword: bool = True,
    max_split: int = 1,
) -> List["WordToken"]:
    """
    Orchestrates the full "prediction round":
      1) Build template string with [blank] markers from tokens
      2) Call `predictor(template_str)` -> fixed sentence
      3) Fill the [blank] slots back into `tokens`
    """
    template_str, blank_idx = tokens_to_template(tokens)

    # If there are no blanks, nothing to do
    if not blank_idx:
        return tokens

    fixed_sentence = predictor(template_str)
    updated = apply_prediction_to_tokens(
        original_tokens=tokens,
        fixed_sentence=fixed_sentence,
        blank_indices=blank_idx,
        split_multiword=split_multiword,
        max_split=max_split,
    )
    return updated