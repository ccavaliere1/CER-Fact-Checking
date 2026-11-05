from pathlib import Path

import faiss
import numpy as np


def load_embeddings(
    embeddings_file,
    doc_ids_file,
    faiss_index_file,
    debug=False,
):
    """
    Carica gli artefatti del retrieval SciFact.

    embeddings_file:
        embedding dei documenti del corpus SciFact.

    doc_ids_file:
        array in cui doc_ids[i] corrisponde al vettore i
        e quindi alla posizione i dell'indice FAISS.

    faiss_index_file:
        indice FAISS costruito sugli stessi embedding.
    """
    embeddings_path = Path(embeddings_file)
    doc_ids_path = Path(doc_ids_file)
    index_path = Path(faiss_index_file)

    missing_files = [
        str(path)
        for path in (embeddings_path, doc_ids_path, index_path)
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "File SciFact mancanti: " + ", ".join(missing_files)
        )

    embeddings = np.load(embeddings_path)
    doc_ids = np.load(doc_ids_path, allow_pickle=False)
    index = faiss.read_index(str(index_path))

    if embeddings.ndim != 2:
        raise ValueError(
            f"Gli embedding devono essere una matrice 2D; "
            f"forma ricevuta: {embeddings.shape}"
        )

    if doc_ids.ndim != 1:
        raise ValueError(
            f"doc_ids deve essere un array 1D; "
            f"forma ricevuta: {doc_ids.shape}"
        )

    if embeddings.shape[0] != len(doc_ids):
        raise ValueError(
            "Numero di embedding e numero di doc_id non coerente: "
            f"{embeddings.shape[0]} != {len(doc_ids)}"
        )

    if index.ntotal != len(doc_ids):
        raise ValueError(
            "Il numero di vettori nell'indice FAISS non coincide con "
            f"il numero di doc_id: {index.ntotal} != {len(doc_ids)}"
        )

    if index.d != embeddings.shape[1]:
        raise ValueError(
            "La dimensione dell'indice FAISS non coincide con quella "
            f"degli embedding: {index.d} != {embeddings.shape[1]}"
        )

    if debug:
        print(
            "Embedding SciFact, doc_id e indice FAISS caricati correttamente."
        )

    return embeddings, doc_ids.astype(np.int64, copy=False), index



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