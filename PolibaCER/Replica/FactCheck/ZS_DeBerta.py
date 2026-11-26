from transformers import pipeline
import logging

# Function to generate justifications
def generate_justification(row, classes=None):
    if classes is None:
        classes = ["True", "False", "NEI"]


    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")

    claim = row['Claim']
    sentences=['Sentence Concatenate']
    justifications = row["Justification"]
    true_label = row['True Label']
    predicted_label = row['Predicted Label']

    justification1 = f'"prediction: {predicted_label}" for: justification used for the prediction "{justifications}" with sentencies {sentences}'
    max_length = 512 # Some buffer for tokens
    if len(justification1) > max_length:
        justification1 = justification1[:max_length]
    output = zeroshot_classifier(claim, classes, hypothesis_template="The claim is '{}' for: " + justification1, multi_label=False)

    final_justification = f'{output}. Predicted label was {predicted_label}. True label was {true_label}'

    return final_justification