import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from PolibaCER.SciFact.DataLoader.DataLoader import build_scifact_index
from PolibaCER.FactCheck.Retriver import retrieve_scifact_top_abstracts
from collections import Counter

corpus_path = Path(__file__).resolve().parents[1] / "data" / "SciFact"
LABELS = ["TRUE", "FALSE", "NEI"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
STANDARD_JUSTIFICATION = "No justification provided."
NEI_JUSTIFICATION = "NEI - Not enough information to evaluate this claim."

SCIFACT_LABELS = {
    "SUPPORT": "TRUE",
    "CONTRADICT": "FALSE",
    "TRUE": "TRUE",
    "FALSE": "FALSE",
    "NEI": "NEI",
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    corpus = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            corpus[str(item["doc_id"])] = item

    return corpus


def evidence_to_text(evidence: dict[str, list[dict[str, Any]]], corpus: dict[str, dict[str, Any]]) -> str:
    sentences = []

    for doc_id, annotations in evidence.items():
        document = corpus.get(str(doc_id))

        if document is None:
            raise KeyError(f"Documento {doc_id} non presente nel corpus SciFact.")

        abstract = document.get("abstract", [])

        for annotation in annotations:
            for sentence_id in annotation.get("sentences", []):
                if sentence_id >= len(abstract):
                    raise IndexError(
                        f"Frase {sentence_id} non valida per il documento {doc_id}."
                    )
                sentences.append(abstract[sentence_id])

    return " ".join(sentences)


def get_label(item: dict[str, Any]) -> str:
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
            f"Il claim {item.get('id')} contiene più label: {sorted(labels)}"
        )

    label = labels.pop()

    if label not in SCIFACT_LABELS:
        raise ValueError(f"Label non riconosciuta: {label}")

    return SCIFACT_LABELS[label]


def load_scifact_examples(claims_path: Path, corpus_path: Path) -> list[dict[str, Any]]:
    corpus = load_corpus(corpus_path)
    examples = []

    with claims_path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            evidence = item.get("evidence") or {}

            examples.append(
                {
                    "id": item["id"],
                    "claim": item["claim"],
                    "evidence": evidence_to_text(evidence, corpus) + " " + item.get("justification", STANDARD_JUSTIFICATION)
                    if evidence
                    else item.get("justification", STANDARD_JUSTIFICATION),
                    "label": get_label(item),
                }
            )

    return examples


class SciFactDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer, max_length: int) -> None:
        self.labels = torch.tensor([LABEL2ID[example["label"]] for example in examples], dtype=torch.long)

        texts = [
            (
                example["claim"],
                example["evidence"] if example["evidence"] else "",
            )
            for example in examples
        ]

        self.encodings = tokenizer(
            [claim for claim, _ in texts],
            [evidence for _, evidence in texts],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


@dataclass
class TrainConfig:
    model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 512
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    validation_size: float = 0.1
    validation_top_k: int = 5
    seed: int = 42


def evaluate(model, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int], float]:

    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []

    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            output = model(**batch)
            total_loss += output.loss.item()
            predictions.extend(output.logits.argmax(dim=-1).cpu().tolist())
            targets.extend(batch["labels"].cpu().tolist())

    macro_f1 = f1_score(
        targets,
        predictions,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0,
    )

    return total_loss / max(len(loader), 1), targets, predictions, macro_f1


def build_retrieved_validation_examples(examples: list[dict[str, Any]], retriever_model, index, doc_ids, documents, top_k: int) -> list[dict[str, Any]]:

    retrieved_examples = []

    for example in examples:
        # if example["label"] == "NEI":
        #     evidence = "No evidence available. Not enough information to evaluate this claim."
        # else:
        retrieved = retrieve_scifact_top_abstracts(
            claim=example["claim"],
            model=retriever_model,
            index=index,
            doc_ids=doc_ids,
            documents=documents,
            top_k=top_k,
        )

        evidence = "\n".join(
            result["abstract"]
            for result in retrieved
        )
        # --- else end ----

        retrieved_examples.append({
            **example,
            "evidence": evidence,
        })


    return retrieved_examples


def save_finetuned_model(model, tokenizer, output_dir: Path, config: TrainConfig) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with (output_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model_name": config.model_name,
                "max_length": config.max_length,
                "labels": LABELS,
            },
            file,
            indent=2,
        )


def load_finetuned_model(model_dir: str | Path, device: str | None = None):

    model_dir = Path(model_dir)
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(selected_device)
    model.eval()

    return model, tokenizer, selected_device



def predict(model, tokenizer, claim: str, justification: str = "", device: torch.device | None = None, max_length: int = 512) -> dict[str, Any]:

    if device is None:
        device = next(model.parameters()).device

    if not justification:
        justification = STANDARD_JUSTIFICATION

    encoded = tokenizer(
        claim,
        justification,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = int(probabilities.argmax())

    return {
        "label": ID2LABEL[predicted_id],
        "confidence": float(probabilities[predicted_id]),
        "probabilities": {
            ID2LABEL[i]: float(probabilities[i])
            for i in range(len(LABELS))
        },
    }


def train(train_examples: list[dict[str, Any]], output_dir: Path, config: TrainConfig) -> None:
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    labels = [example["label"] for example in train_examples]

    train_data, validation_data = train_test_split(
        train_examples,
        test_size=config.validation_size,
        random_state=config.seed,
        stratify=labels,
    )

    validation_retriever = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    documents, doc_ids, retrieval_index = build_scifact_index(model=validation_retriever, data_path=corpus_path, NEW_INDEX=False)
    retrieved_validation_data = build_retrieved_validation_examples(
        validation_data,
        validation_retriever,
        retrieval_index,
        doc_ids,
        documents,
        config.validation_top_k,
    )

    train_dataset = SciFactDataset(train_data, tokenizer, config.max_length)
    validation_dataset = SciFactDataset(retrieved_validation_data, tokenizer, config.max_length)

    print(
        "Validation class distribution:",
        Counter(example["label"] for example in retrieved_validation_data),
    )
    print(
        "Validation examples:",
        len(retrieved_validation_data),
    )


    class_counts = torch.bincount(train_dataset.labels, minlength=len(LABELS)).long()
    max_class_count = int(class_counts.max().item())
    balanced_indices = []
    rng = np.random.default_rng(config.seed)

    for class_id in range(len(LABELS)):
        class_indices = np.flatnonzero(
            train_dataset.labels.numpy() == class_id
        )

        if len(class_indices) == 0:
            raise ValueError(
                f"La classe {ID2LABEL[class_id]} non contiene esempi nel train set."
            )

        sampled_indices = rng.choice(
            class_indices,
            size=max_class_count,
            replace=True,
        )
        balanced_indices.extend(sampled_indices.tolist())

    rng.shuffle(balanced_indices)
    train_sampler = SubsetRandomSampler(balanced_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_weights = torch.ones(
        len(LABELS),
        dtype=torch.float32,
        device=device,
    )

    print("Train class distribution:", {
        ID2LABEL[index]: int(class_counts[index])
        for index in range(len(LABELS))
    })
    print("Balanced samples per class:", max_class_count)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_validation_f1 = -1.0

    for epoch in range(config.epochs):
        model.train()

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", unit="batch", leave=False, dynamic_ncols=True) as progress:
            for batch in progress:
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                optimizer.zero_grad(set_to_none=True)

                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                )

                loss = torch.nn.functional.cross_entropy(
                    output.logits,
                    batch["labels"],
                    weight=class_weights,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        validation_loss, y_true, y_pred, validation_f1 = evaluate(model, validation_loader, device)

        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"validation loss: {validation_loss:.4f}"
        )
        print(f"Validation macro-F1: {validation_f1:.4f}")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=list(range(len(LABELS))),
                target_names=LABELS,
                zero_division=0,
            )
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            save_finetuned_model(model, tokenizer, output_dir, config)

    print(f"Modello salvato in: {output_dir}")


def main() -> None:
    config = TrainConfig(
        model_name="microsoft/deberta-v3-large",
        epochs=8,
        batch_size=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=1e-5,
        max_length=512,
        seed=42,
    ) # Epochs: 4, Batch size: 4, Learning rate: 2e-5, Max length: 512, Seed: 42 -> 0.5986


    train_examples = load_scifact_examples(
        Path(__file__).resolve().parents[1]/"data"/"SciFact" / "claims_train_all_justified.jsonl",
        Path(__file__).resolve().parents[1]/"data"/"SciFact" / "corpus.jsonl",
    )

    train(train_examples, Path("models") / "cer-deberta-v3-large-scifact", config)


if __name__ == "__main__":
    main()