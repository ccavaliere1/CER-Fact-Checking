import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PolibaCER.SciFact.DataLoader.DataLoader import build_scifact_index
from PolibaCER.FactCheck.Retriver import retrieve_scifact_top_abstracts


def load_test_claims(path: Path, limit: int | None):
    with path.open("r", encoding="utf-8") as file:
        claims = [json.loads(line) for line in file]

    return claims[:limit] if limit else claims


def evaluate(data_dir: Path, top_k: int, limit: int | None):
    retriever_model = SentenceTransformer("all-MiniLM-L6-v2")

    documents, doc_ids, index = build_scifact_index(
        model=retriever_model,
        data_path=data_dir,
        NEW_INDEX=False,
    )

    claims = load_test_claims(
        data_dir / "claims_test.jsonl",
        limit,
    )

    recalls = []
    hits = []
    reciprocal_ranks = []
    errors = []

    for item in claims:
        claim = item["claim"]
        gold_ids = {
            int(doc_id)
            for doc_id in item.get("cited_doc_ids", [])
        }

        retrieved = retrieve_scifact_top_abstracts(
            claim=claim,
            model=retriever_model,
            index=index,
            doc_ids=doc_ids,
            documents=documents,
            top_k=top_k,
        )

        retrieved_ids = [
            int(result["doc_id"])
            for result in retrieved
        ]

        found_ids = gold_ids.intersection(retrieved_ids)

        recall = (
            len(found_ids) / len(gold_ids)
            if gold_ids
            else 0.0
        )

        hit = 1.0 if found_ids else 0.0

        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in gold_ids:
                reciprocal_rank = 1.0 / rank
                break

        recalls.append(recall)
        hits.append(hit)
        reciprocal_ranks.append(reciprocal_rank)

        if not found_ids:
            errors.append(
                {
                    "claim": claim,
                    "gold": sorted(gold_ids),
                    "retrieved": retrieved_ids,
                }
            )

    total = len(claims)

    print("\nPrimi errori:")
    for error in errors[:10]:
        print(f"\nClaim: {error['claim']}")
        print(f"Gold: {error['gold']}")
        print(f"Retrieved: {error['retrieved']}")

    print(f"\nClaims valutati: {total}")
    print(f"Recall@{top_k}: {sum(recalls) / total:.4f}")
    print(f"Hit@{top_k}: {sum(hits) / total:.4f}")
    print(f"MRR@{top_k}: {sum(reciprocal_ranks) / total:.4f}")


def main():
    evaluate(
        data_dir=Path(__file__).resolve().parents[2] / "data" / "SciFact",
        top_k=20,
        limit=None,
    )


if __name__ == "__main__":
    main()