from pathlib import Path
from sentence_transformers import SentenceTransformer
from PolibaCER.ScientificEvidence.CorpusLoader import build_scifact_index
from PolibaCER.ScientificEvidence.Retriver import retrieve_top_abstracts
from ClaimDetection import input_claim_text

DATA_PATH = Path(__file__).resolve().parent / "data" / "SciFact"
MODEL = SentenceTransformer("all-MiniLM-L6-v2")


if __name__ == "__main__":

    documents, doc_ids, index = build_scifact_index(
        model=MODEL,
        data_path=DATA_PATH,
        NEW_INDEX=False,
    )

    claim = input_claim_text()


    top_abstracts = retrieve_top_abstracts(
        claim=claim,
        model=MODEL,
        index=index,
        doc_ids=doc_ids,
        documents=documents,
        top_k=5,
    )

    # PRINT RESULTS
    for rank, result in enumerate(top_abstracts, start=1):
        print(f"\n[{rank}] doc_id={result['doc_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Titolo: {result['title']}")
        print(f"Abstract: {result['abstract']}")


