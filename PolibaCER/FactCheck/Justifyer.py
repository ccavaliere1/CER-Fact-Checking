import re


def parse_response(response):
    """
    Parse the response from the model and extract the fields.

    Args:
        response (str): Response string from the model.

    Returns:
        tuple: Extracted fields from the response.
    """
    # Initial values for the fields
    first_label = "Non trovato"
    justification = "Non trovato"
    supporting = "Non trovato"
    refusing = "Non trovato"
    notes = "Non trovato"

    # Regular expression patterns for extracting fields
    patterns = {
        "first_label": r"(?:Label|Etichetta):\s*(.*?)\n",
        "justification": r"(?:Justification|Giustificazione):\s*(.*?)(?=\n(?:Supporting sentences|Frasi di supporto|Refusing sentences|Frasi di rifiuto))",
        "supporting": r"(?:Supporting sentences from abstracts|Frasi di supporto dai documenti):\n(.*?)(?=\n(?:Refusing sentences|Frasi di rifiuto))",
        "refusing": r"(?:Refusing sentences from abstracts|Frasi di rifiuto dai documenti):\n(.*?)(?=\n(?:Note|Nota):|$)",
        "notes": r"(?:Note|Nota):\s*(.*)"
    }

    # Extract the fields using regular expressions
    if match := re.search(patterns["first_label"], response, re.DOTALL | re.IGNORECASE):
        first_label = match.group(1).strip()
    if match := re.search(patterns["justification"], response, re.DOTALL | re.IGNORECASE):
        justification = match.group(1).strip()
    if match := re.search(patterns["supporting"], response, re.DOTALL | re.IGNORECASE):
        supporting = [{"text": sentence.strip(), "abstract": f"abstract_{i + 1}"} for i, sentence in
                      enumerate(match.group(1).strip().split('\n')) if sentence.strip()]
    if match := re.search(patterns["refusing"], response, re.DOTALL | re.IGNORECASE):
        refusing = [{"text": sentence.strip(), "abstract": f"abstract_{i + 1}"} for i, sentence in
                    enumerate(match.group(1).strip().split('\n')) if sentence.strip()]
    if match := re.search(patterns["notes"], response, re.DOTALL | re.IGNORECASE):
        notes = match.group(1).strip()

    # Return the extracted fields
    return first_label, justification, supporting, refusing, notes



def generate_justification(zeroshot_classifier, query, justification):
    """
    Generate a justification for the claim using the Zero-Shot Classification model.

    Args:
        zeroshot_classifier:
        query (str): Claim to fact-check.
        justification (str): Justification for the claim.

    Returns:
        str: Final justification for the claim.
    """
    # Define the classes for the Zero-Shot Classification model
    Class = ["True", "False", "NEI"]

    # Generate the justification text
    justification_text = (
        f'Justification: "{justification}"'
    )

    # Limit the justification text to a maximum length
    max_length = 512
    if len(justification_text) > max_length:
        justification_text = justification_text[:max_length]

    # Generate the final justification using the Zero-Shot Classification model
    output = zeroshot_classifier(
        query,
        Class,
        hypothesis_template=f"The claim is '{{}}' for: {justification_text}",
        multi_label=False
    )

    # Prepare the final justification text
    final_justification = f'{output}.'

    return final_justification