import json
from pathlib import Path
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline
from PolibaCER.SciFactTraning.traning_model import load_finetuned_model, predict

MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
LABELS = ["SUPPORT", "CONTRADICT", "NEI"]
BASE_PATH = Path(__file__).resolve().parents[2]
model_dir = BASE_PATH / "models" / "cer-deberta-v3-large-scifact"
TRAINED_CLASSIFIER_MODEL, tokenizer, device = load_finetuned_model(model_dir)


def load_corpus(path: Path) -> dict[str, dict]:
    corpus = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            corpus[str(item["doc_id"])] = item

    return corpus


def evidence_to_text(evidence: dict, corpus: dict[str, dict]) -> str:
    sentences = []

    for doc_id, annotations in evidence.items():
        document = corpus[str(doc_id)]

        for annotation in annotations:
            for sentence_id in annotation.get("sentences", []):
                sentences.append(document["abstract"][sentence_id])

    return " ".join(sentences)


def get_gold_label(item: dict) -> str:
    evidence = item.get("evidence") or {}

    if not evidence:
        return "NEI"

    labels = {
        annotation["label"].upper()
        for annotations in evidence.values()
        for annotation in annotations
    }

    if len(labels) != 1:
        raise ValueError(
            f"Label multiple per claim {item['id']}: {labels}"
        )

    return labels.pop()


def evaluate(data_dir: Path, device: str | None, limit: int | None, FN=True):
    selected_device = (
        0
        if device == "cuda" or (device is None and torch.cuda.is_available())
        else -1
    )

    corpus = load_corpus(data_dir / "corpus.jsonl")
    with (data_dir / "claims_test.jsonl").open("r",encoding="utf-8") as file:
        claims = [json.loads(line) for line in file]
    if limit is not None:
        claims = claims[:limit]

    y_true = []
    y_pred = []

    for index, item in enumerate(claims, start=1):
        claim = item["claim"]
        evidence = evidence_to_text(
            item.get("evidence") or {},
            corpus,
        )

        if not evidence:
            evidence = "No evidence is available for this claim. No scientific evidence supports or contradicts this claim."

        if FN:
            result = predict(
                model=TRAINED_CLASSIFIER_MODEL,
                tokenizer=tokenizer,
                claim=claim,
                justification=(
                    "The claim is {} according to this scientific evidence: "
                    + evidence
                ),
                device=device,
            )

            model_label_mapping = {
                "TRUE": "SUPPORT",
                "FALSE": "CONTRADICT",
                "NEI": "NEI",
            }

            predicted_label = model_label_mapping[result["label"]]
            confidence = result["confidence"]
            true_label = get_gold_label(item)

        else:
            classifier = pipeline(
                task="zero-shot-classification",
                model=MODEL_NAME,
                device=selected_device,
            )

            result = classifier(
                sequences=claim,
                candidate_labels=LABELS,
                hypothesis_template=(
                    "The claim is {} according to this scientific evidence: "
                    + evidence
                ),
                multi_label=False,
            )

            predicted_label = result["labels"][0]
            confidence = result["scores"][0]
            true_label = get_gold_label(item)


        y_true.append(true_label)
        y_pred.append(predicted_label)

        print(
            f"[{index}/{len(claims)}] "
            f"Predicted: {predicted_label} ({confidence:.4f}) | "
            f"Gold: {true_label} | "
            f"{claim}"
        )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print("Labels:", LABELS)
    print(confusion_matrix(y_true, y_pred, labels=LABELS))


def main():
    #evaluate(data_dir=Path(__file__).resolve().parents[2] / "data" / "SciFact", device=device, limit=None, FN=False)
    evaluate(data_dir=Path(__file__).resolve().parents[2] / "data" / "SciFact", device=device, limit=None, FN=True)

if __name__ == "__main__":
    main()