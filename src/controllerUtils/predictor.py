from openai import OpenAI
from dotenv import load_dotenv
import os



def word_predictor(transcription: str):
    load_dotenv(dotenv_path="config.env")

    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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


    # Call GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The transcription:\n\n{transcription}"}
        ],
        temperature=0.2  # low temperature for more deterministic output
    )

    # return the result
    return response
