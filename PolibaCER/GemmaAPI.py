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

    def standard_response():
        return (
            "Label: NEI\n"
            "Justification: Impossibile ottenere una risposta dal modello.\n"
            "Supporting sentences from abstracts:\n"
            "Refusing sentences from abstracts:\n"
            "Note: Google AI non disponibile."
        )


    answer = ""
    max_attempts = 10
    base_backoff = 2
    max_backoff = 30
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
                warnings.warn(
                    f"Errore Google non recuperabile: {error}",
                    RuntimeWarning,
                )
                return standard_response()

            if attempt == max_attempts - 1:
                answer = standard_response()

                warnings.warn(
                    "Tentativi esauriti. Restituita una risposta predefinita.",
                    RuntimeWarning,
                )
                break

            backoff_time = min(base_backoff * (2 ** attempt), max_backoff)

            warnings.warn(
                f"Errore Google 500. Nuovo tentativo "
                f"({attempt + 2}/{max_attempts})...",
                RuntimeWarning,
            )
            time.sleep(backoff_time)

        except Exception as error:
            warnings.warn(
                f"Errore durante la richiesta a Google: {error}. "
                "Restituita una risposta standard.",
                RuntimeWarning,
            )
            return standard_response()

    time.perf_counter()
    # print(answer)
    return answer

