import argparse
import json
from collections import Counter
from pathlib import Path


LABELS = ("SUPPORT", "CONTRADICT", "NEI")


def get_label(item: dict) -> str:
    evidence = item.get("evidence") or {}

    if not evidence:
        return "NEI"

    labels = {
        annotation.get("label", "").strip().upper()
        for annotations in evidence.values()
        for annotation in annotations
        if annotation.get("label")
    }

    if not labels:
        return "NEI"

    if len(labels) != 1:
        raise ValueError(
            f"Il claim {item.get('id')} contiene label diverse: {sorted(labels)}"
        )

    label = labels.pop()

    if label not in {"SUPPORT", "CONTRADICT"}:
        raise ValueError(
            f"Label non riconosciuta nel claim {item.get('id')}: {label}"
        )

    return label


def count_labels(path: Path) -> Counter[str]:
    counts = Counter({label: 0 for label in LABELS})

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON non valido in {path}, riga {line_number}"
                ) from error

            counts[get_label(item)] += 1

    return counts


def print_counts(name: str, counts: Counter[str]) -> None:
    total = sum(counts.values())

    print(f"\n{name}")
    print(f"Totale: {total}")

    for label in LABELS:
        print(f"{label}: {counts[label]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conta SUPPORT, CONTRADICT e NEI nei dataset SciFact."
    )

    default_data_dir = (
        Path(__file__).resolve().parents[1] / "data" / "SciFact"
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=default_data_dir / "claims_train.jsonl",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=default_data_dir / "claims_test.jsonl",
    )

    args = parser.parse_args()

    train_counts = count_labels(args.train)
    test_counts = count_labels(args.test)

    print_counts("claims_train.jsonl", train_counts)
    print_counts("claims_test.jsonl", test_counts)


if __name__ == "__main__":
    main()