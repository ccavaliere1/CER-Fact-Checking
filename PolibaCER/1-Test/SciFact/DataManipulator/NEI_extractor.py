import argparse
import json
from pathlib import Path


def extract_nei_examples(input_path: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON non valido alla riga {line_number}"
                ) from error

            if not item.get("evidence"):
                output_file.write(
                    json.dumps(item, ensure_ascii=False) + "\n"
                )
                count += 1

    return count


FAILED_JUSTIFICATION = (
    "Impossibile ottenere una risposta dal modello."
)


def extract_failed_justifications(
    input_path: Path,
    output_path: Path,
) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON non valido alla riga {line_number}"
                ) from error

            if item.get("justification") == FAILED_JUSTIFICATION:
                output_file.write(
                    json.dumps(item, ensure_ascii=False) + "\n"
                )
                count += 1

    return count

def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON non valido in {path}, riga {line_number}"
                ) from error

    return records


def merge_justifications(
    first_path: Path,
    second_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """
    Unisce i due JSONL per id.

    Le justification presenti nel secondo file sostituiscono quelle
    presenti nel primo file.
    """
    first_records = load_jsonl(first_path)
    second_records = load_jsonl(second_path)

    records_by_id = {
        int(record["id"]): record
        for record in first_records
    }

    replaced = 0
    added = 0

    for record in second_records:
        record_id = int(record["id"])

        if record_id in records_by_id:
            records_by_id[record_id]["justification"] = record.get(
                "justification"
            )
            replaced += 1
        else:
            records_by_id[record_id] = record
            added += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record_id in sorted(records_by_id):
            file.write(
                json.dumps(
                    records_by_id[record_id],
                    ensure_ascii=False,
                )
                + "\n"
            )

    return replaced, added

def add_justifications_to_test(
    merged_justifications_path: Path,
    test_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """
    Inserisce la justification nei record claims_test con lo stesso id.
    """
    justification_records = load_jsonl(merged_justifications_path)
    test_records = load_jsonl(test_path)

    justifications = {
        int(record["id"]): record["justification"]
        for record in justification_records
        if record.get("justification")
    }

    matched = 0

    for record in test_records:
        record_id = int(record["id"])

        if record_id in justifications:
            record["justification"] = justifications[record_id]
            matched += 1
        else:
            record.pop("justification", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in test_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    unmatched = len(justifications) - matched
    return matched, unmatched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estrae i claim NEI da claims_train.jsonl."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "data"
            / "SciFact"
            / "claims_train.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "data"
            / "SciFact"
            / "claims_train_nei.jsonl"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------- EXTRACT NEI ------------------------------------------- #
    # count = extract_nei_examples(args.input, args.output)
    #
    # print(f"Record NEI estratti: {count}")
    # print(f"File creato: {args.output}")

    # ----------------------------------------- EXTRACT FAILED JUSTIFICATIONS ----------------------------------------- #
    # data_dir = Path(__file__).resolve().parents[1] / "data" / "SciFact"
    #
    # count = extract_failed_justifications(
    #     input_path=data_dir / "claims_train_nei_mj50.jsonl",
    #     index_output_path=data_dir / "claims_train_failed_justifications.jsonl",
    # )
    # print(f"Record estratti: {count}")


    # ----------------------------------------- MERGE JUSTIFICATIONS ----------------------------------------- #
    data_dir = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "SciFact"
    )

    # replaced, added = merge_justifications(
    #     first_path=data_dir / "+304-50.jsonl",
    #     second_path=data_dir / "+50.jsonl",
    #     index_output_path=data_dir / "304_merged_justifications.jsonl",
    # )

    # print(f"Justification sostituite: {replaced}")
    # print(f"Nuovi claim aggiunti: {added}")

    matched, unmatched = add_justifications_to_test(
        merged_justifications_path=data_dir / "claims_train_justifications_processed.jsonl",
        test_path=data_dir / "claims_train.jsonl",
        output_path=data_dir / "claims_train_with_justifications_evidence.jsonl",
    )

    print(f"Claim test aggiornati: {matched}")
    print(f"Justification train senza corrispondenza nel test: {unmatched}")


if __name__ == "__main__":
    main()