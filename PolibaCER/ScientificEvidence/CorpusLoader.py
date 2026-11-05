import json
from pathlib import Path
import faiss
import numpy as np

def load_scifact_corpus(corpus_path):
    """
    Carica il corpus SciFact.

    Restituisce:
        documents: dizionario doc_id -> dati del documento
        doc_ids: array degli ID nell'ordine usato dall'indice FAISS
    """
    documents = {}

    with Path(corpus_path).open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            doc_id = int(record["doc_id"])
            sentences = record["abstract"]

            documents[doc_id] = {
                "doc_id": doc_id,
                "title": record["title"],
                "sentences": sentences,
                "text": record["title"] + " " + " ".join(sentences),
            }

    doc_ids = np.array(list(documents.keys()), dtype=np.int64)
    return documents, doc_ids


def build_scifact_index(corpus_path, model):
    """
    Costruisce un indice FAISS sul corpus SciFact.

    La similarità utilizzata è il prodotto interno tra vettori normalizzati, equivalente alla cosine similarity.
    """
    documents, doc_ids = load_scifact_corpus(corpus_path)

    texts = [documents[int(doc_id)]["text"] for doc_id in doc_ids]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return documents, doc_ids, index