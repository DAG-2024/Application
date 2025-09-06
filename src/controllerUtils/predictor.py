from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import sys
import logging
from pathlib import Path

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
app_logger = logging.getLogger("controller_app")
app_logger.setLevel(logging.DEBUG)

if not app_logger.handlers:
    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    app_logger.addHandler(file_handler)
    app_logger.addHandler(stream_handler)

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
                          - Predict exactly one word per `[blank]` placeholder.
                          - However, you may predict two words if the first word is an indefinite or definite article, such as `"a"`, `"an"`, or `"the"` (e.g., `"a cat"`, `"the tree"`).
                          - Ignore any punctuation that appear near the `[blank]` placeholder. It may be incorrect due to audio corruption. Rely on sentence meaning and grammar to make your decision.
                          - Preserve the rest of the transcription exactly as it is, replacing only the `[blank]` placeholders with the predicted word, as a result, the number of words in the result should be almost exactly the same as the original. 
                          - Your output should be a single string — the full transcription with predicted words inserted in place of the `[blank]` placeholders.
                          - use the entire transcription as a strong reference and the transcription's subject to determine the missing word.
                          - Do not add any additional commentary or explanation; only return the completed transcription.
                          ---

                          ### Examples (for format and reasoning only):

                          **Input:**  
                          "The tree is. [blank] Due to its height, it gets a lot of sunlight."

                          **Output:**  
                          "The tree is tall. Due to its height, it gets a lot of sunlight."
                          
                          **Input:**  
                          "Now, all three of these are tied together by Ohm’s Law.
                           Ohm’s Law is V = I × R. If you increase the [blank] you push more current, unless resistance is high.
                           If you [blank] the resistance, then for the same voltage, less current flows."
                            
                          **Output:**  
                            "Now, all three of these are tied together by Ohm’s Law.
                            Ohm’s Law is V = I × R. If you increase the voltage, you push more current, unless resistance is high.
                            If you increase the resistance, then for the same voltage, less current flows."
                      
                          ###
                       """
                     )

    # IMPORTANT: `model` is your DEPLOYMENT NAME for gpt-35-turbo in Azure
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The transcription:\n\n{transcription}"}
        ],
        temperature=0.1
    )
    # If you want the full SDK response object:
    # return response

    # If you just want the completed transcript string:

    return response.choices[0].message.content
