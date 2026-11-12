import json
import os
from pathlib import Path
import faiss
import numpy as np

# ----------------------------------------------- PUBMED EMBEDDINGS LOADER ------------------------------------------------ #
def load_pubmed_embeddings(embeddings_file, pmid_file, faiss_index_file, debug=False):
    """
    Load embeddings, PMIDs, and FAISS index from the specified files.

    Args:
        embeddings_file (str): File path for the embeddings.
        pmid_file (str): File path for the PMIDs.
        faiss_index_file (str): File path for the FAISS index.

    Returns:
        tuple: Tuple containing the embeddings, PMIDs, and FAISS index.
    """
    # Check if the files exist
    if not (os.path.exists(embeddings_file) and os.path.exists(pmid_file) and os.path.exists(faiss_index_file)):
        raise FileNotFoundError("One or more files not found. Please check the file paths.")

    # Load the embeddings and PMIDs
    embeddings = np.load(embeddings_file)
    pmids = np.load(pmid_file, allow_pickle=True)

    # Load the FAISS index
    index = faiss.read_index(faiss_index_file)

    return embeddings, pmids, index
# ----------------------------------------------- END PUBMED EMBEDDINGS LOADER ------------------------------------------------ #



# ----------------------------------------------- SCIFACT CORPUS LOADER ------------------------------------------------ #
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

def build_scifact_index(model, data_path, NEW_INDEX=False):
    """
    Costruisce un indice FAISS sul corpus SciFact.

    La similarità utilizzata è il prodotto interno tra vettori normalizzati, equivalente alla cosine similarity.
    """

    # Se tutti i file esistono, carica gli embedding e l'indice FAISS
    corpus_path = Path(data_path) / "corpus.jsonl"
    faiss_index_path = Path(data_path) / "faiss_index.index"

    # Carica il corpus SciFact
    documents, doc_ids = load_scifact_corpus(corpus_path)
    texts = [documents[int(doc_id)]["text"] for doc_id in doc_ids]



    if faiss_index_path.is_file() and not NEW_INDEX:
        print("Caricamento degli embedding e dell'indice FAISS esistenti...")
        index = load_index(data_path)
    else:
        print("Costruzione di nuovi embedding e indice FAISS...")

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # Salva gli embedding, i doc_id e l'indice FAISS
        save_index(index, data_path)


    return documents, doc_ids, index

# ----------------------------------------------- END SCIFACT CORPUS LOADER ----------------------------------------------- #

# ----------------------------------------------- SCIFACT INDEX UTILS ------------------------------------------------ #
def save_index(index, data_path):
    faiss_index_file = Path(data_path) / "faiss_index.index"
    faiss.write_index(index, str(faiss_index_file))

def load_index(data_path):

    faiss_index_file = Path(data_path) / "faiss_index.index"

    if not faiss_index_file.is_file():
        raise FileNotFoundError(f"Indice FAISS non trovato: {faiss_index_file}")

    index = faiss.read_index(str(faiss_index_file))
    return index