import re

def extract_label_and_score(result):
    """
    Extracts the predicted label and score from the result string.

    Args:
        result (str): String containing the prediction result.

    Returns:
        tuple: Predicted label and score.
    """
    # Extract the predicted label
    label_match = re.search(r"'labels': \['(.*?)'", result)
    predicted_label = label_match.group(1) if label_match else None

    # Extract the score
    score_match = re.search(r"'scores': \[(\d+\.\d+)", result)
    score_label = float(score_match.group(1)) if score_match else None

    return predicted_label, score_label


def clean_phrases(phrases, pattern):
    """
    Clean and extract phrases from a list of strings using a specified pattern.

    Args:
        phrases (list): List of strings containing phrases.
        pattern (str): Regular expression pattern to extract phrases.

    Returns:
        list: List of cleaned phrases as dictionaries with text and abstract keys
    """
    cleaned_phrases = []

    for phrase in phrases:
        matches = re.findall(pattern, phrase)
        cleaned_phrases.extend([{"text": match[0], "abstract": f"abstract_{match[1]}"} for match in matches])

    return cleaned_phrases


def extract_supporting_and_refusing_sentences(main_global, top_abstracts, supporting, refusing):
    # Extract the abstracts and references
    abstracts = {}
    for i in range(1, len(top_abstracts) + 1):
        abstracts[f"abstract_{i}"] = main_global[f"abstract_{i}"]

    pattern = r'-\s*[\"\u201c\u201d]?\s*(.*?)\s*[\"\u201c\u201d]?\s*\([Aa]bstract[_\s]*(\d+)\)'

    supporting_texts = []
    for item in supporting:
        try:
            supporting_texts.append(item["text"])
        except (TypeError, KeyError):
            continue
    supporting = clean_phrases(supporting_texts, pattern)

    refusing_text = []
    for item in refusing:
        try:
            refusing_text.append(item["text"])
        except (TypeError, KeyError):
            continue
    refusing = clean_phrases(refusing_text, pattern)

    return abstracts, supporting, refusing

def proccess_abstracts(abstracts, supporting, refusing):
    """
    Process the abstracts to highlight supporting and refusing phrases.

    Args:
        abstracts (dict): Dictionary of abstracts.
        supporting (list): List of supporting phrases.
        refusing (list): List of refusing phrases.
    Returns:
        processed_abstracts: Processed abstracts with highlighted phrases
    """
    # Print the final results
    processed_abstracts = {}
    refusing_matches = []
    for abstract_name, abstract_text in abstracts.items():
        # Highlight supporting phrases in green
        supporting_matches = [phrase for phrase in supporting if phrase["abstract"] == abstract_name]

        # Highlight refusing phrases in red
        refusing_matches = [phrase for phrase in refusing if phrase["abstract"] == abstract_name]

        # Add only if supporting matches are found
        if supporting_matches:
            # Add the reference if a corresponding variable exists
            reference_variable = f"reference_{abstract_name.split('_')[1]}"
            if reference_variable in globals():
                reference_value = globals()[reference_variable]
                abstract_text += f"Reference: {reference_value}"

            # Add the processed abstract
            processed_abstracts[abstract_name] = abstract_text

    return processed_abstracts, refusing_matches