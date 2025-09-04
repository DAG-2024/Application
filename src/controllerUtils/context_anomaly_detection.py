from openai import AzureOpenAI
from dotenv import load_dotenv
import os

def ctx_anomaly_detector(transcription: str, indexed_transcription: str):
    load_dotenv(dotenv_path="src/AZURE_OPENAI.env")

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

    # The system prompt to set context
    system_prompt = ("""You are an expert language model helping to identify contextually incorrect words in a computer science lecture transcription.
                      The original audio contained segments of noise that may have masked a single word per segment,
                      potentially causing one incorrect word per sentence.

                        You will be given:
                        1. A clean transcription (plain text).
                        2. A version of the same transcription where each word is paired with its index.

                        Your task is to find **at most one contextually incorrect word per sentence**,
                        based on domain knowledge linguistic and contextual consistency, and return **only its index** — no quotes, no words, no explanation.

                        Important:
                        - Each word's index corresponds to its position in the indexed version, where words are separated by spaces.
                        - Return only the **numerical index** of each word you identify as contextually incorrect, separated by commas if more than one.

                        Examples below are **for output format only** — do not use their content to guide correctness decisions.

                        ---

                        Example 1 (format only):  
                        Transcription: the compiler generates orange for every loop it detects inside chocolate blocks.  
                        Indexed: 0: the 1: compiler 2: generates 3: orange 4: for 5: every 6: loop 7: it 8: detects 9: inside 10: chocolate 11: blocks.  
                        Output: 3, 10

                        Example 2 (format only):  
                        Transcription: algorithms operate on turtles and giraffes to optimize memory usage.  
                        Indexed: 0: algorithms 1: operate 2: on 3: turtles 4: and 5: giraffes 6: to 7: optimize 8: memory 9: usage.  
                        Output: 3, 5

                        ---

                        Now, analyze the transcription below using both the plain and indexed versions.  
                        Return only the indices of the incorrect words.
                     """
    )

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The transcription:\n\n{transcription}" +  f"the indexed transcription: {indexed_transcription}"}
        ],
        temperature=0.2
    )

    # return the result

    return response
