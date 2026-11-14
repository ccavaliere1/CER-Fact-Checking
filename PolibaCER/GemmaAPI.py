import os
import time
import warnings

from google import genai
from google.genai import errors, types

def gemma_generate(input_text):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemma-4-26b-a4b-it"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH",
        )
    )

    answer = ""
    max_attempts = 3
    start_time = time.perf_counter()
    for attempt in range(max_attempts):
        try:
            for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):

                elapsed = time.perf_counter() - start_time
                print(f"\rTempo trascorso: {elapsed:.1f} secondi", end="", flush=True)

                if text := chunk.text:
                    answer += text

            break  # Exit the loop if successful


        except errors.ServerError as error:

            if error.code != 500:
                raise

            if attempt == max_attempts - 1:
                answer = (
                    "Label: NEI\n"
                    "Justification: Impossibile ottenere una risposta dal modello "
                    "dopo i tentativi disponibili.\n"
                    "Supporting sentences from abstracts:\n"
                    "Refusing sentences from abstracts:\n"
                    "Note: Google AI non disponibile."
                )

                warnings.warn(
                    "Tentativi esauriti. Restituita una risposta predefinita.",
                    RuntimeWarning,
                )
                break

            warnings.warn(
                f"Errore Google 500. Nuovo tentativo "
                f"({attempt + 2}/{max_attempts})...",
                RuntimeWarning,
            )
            time.sleep(2)

    time.perf_counter()
    # print(answer)
    return answer