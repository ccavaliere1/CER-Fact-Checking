from sentence_transformers import SentenceTransformer
from PolibaCER.ScientificEvidence.CorpusLoader import build_scifact_index
from PolibaCER.ScientificEvidence.Retriver import retrieve_top_abstracts
from ClaimDetection import input_claim_text

CORPUS_PATH = r"data\SciFact - data\corpus.jsonl"
MODEL = SentenceTransformer("all-MiniLM-L6-v2")


if __name__ == "__main__":

    documents, doc_ids, index = build_scifact_index(
        corpus_path=CORPUS_PATH,
        model=MODEL,
    )

    claim = input_claim_text()




    results = retrieve_top_abstracts(
        claim=claim,
        model=MODEL,
        index=index,
        doc_ids=doc_ids,
        documents=documents,
        top_k=5,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\n[{rank}] doc_id={result['doc_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Titolo: {result['title']}")
        print(f"Abstract: {result['abstract']}")