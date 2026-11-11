from pathlib import Path

import faiss
import numpy as np

def retrieve_top_abstracts(claim, model, index, doc_ids, documents,top_k=5,):
    """
    Recupera i top-k documenti SciFact per un claim.

    Restituisce una lista ordinata per similarità decrescente.
    """
    if not claim or not claim.strip():
        raise ValueError("Il claim non può essere vuoto.")

    if top_k < 1:
        raise ValueError("top_k deve essere almeno 1.")

    # AI VER
    # query_embedding = model.encode(
    #     [claim],
    #     convert_to_numpy=True,
    #     normalize_embeddings=True,
    # ).astype("float32")
    #
    # number_of_documents = index.ntotal
    # effective_top_k = min(top_k, number_of_documents)
    #
    # scores, positions = index.search(query_embedding, effective_top_k)


    # APP VERSION
    query_embedding = model.encode([claim])
    faiss.normalize_L2(query_embedding)  # Normalize the claim embedding (with L2 norm)
    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, position in zip(scores[0], indices[0]):
        doc_id = int(doc_ids[position])
        document = documents[doc_id]

        results.append(
            {
                "doc_id": doc_id,
                "title": document["title"],
                "abstract": document["text"],
                "sentences": document["sentences"],
                "score": float(score),
            }
        )

    return results