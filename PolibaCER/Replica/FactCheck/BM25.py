import json
import sys
import numpy as np
from collections import Counter
import csv
from pathlib import Path
import re
import time
import dill as pickle
import nltk
from huggingface_hub.utils import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from rank_bm25 import BM25Okapi

DATA_PATH = Path(__file__).resolve().parents[1] / "data"
corpus_path = DATA_PATH / "PubMed_p205.csv"
index_output_path = DATA_PATH / "PubMed_BM25Okapi.pkl"
MIN_DF = 10

# Scarica le stopword inglesi se non già presenti in locale
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk_data_dir = Path(sys.prefix) / "nltk_data"
    nltk.download("stopwords", download_dir=str(nltk_data_dir), quiet=True)
    nltk.data.path.insert(0, str(nltk_data_dir))
    STOPWORDS = set(stopwords.words("english"))

stemmer = SnowballStemmer("english")

def preprocess_and_tokenize(text: str) -> list[str]:
    # Replica le normalizzazioni di SparseRetriever + stopwords + stemming
    text = text or ""
    text = text.lower()  # Lowercasing
    text = text.replace("&", " and ")  # Ampersand & acronimi (es. U.S.A. -> usa)
    text = re.sub(r"\b([a-z])\.(?:([a-z])\.)+", lambda m: m.group(0).replace(".", ""), text)
    # Normalizzazione caratteri speciali / rimozione punteggiatura
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Whitespace tokenization, stopwords, min-length e stemming
    tokens: list[str] = []
    for tok in text.split(" "):
        if not tok or tok in STOPWORDS or len(tok) < 2:
            continue
        tokens.append(stemmer.stem(tok))
    return tokens

# -------------------------------------------------- BUILD BM25 INDEX -------------------------------------------------
def build_bm25_index(corpus_path: Path, min_df: int = 10) -> dict:

    start = time.time()

    doc_ids: list[str] = []
    raw_tokenized_docs: list[list[str]] = []
    df_counter: Counter[str] = Counter()

    with open(corpus_path, mode="r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in tqdm(enumerate(reader), desc="Indicizzazione PubMed", unit="doc"):
            doc_id = str(row["PMID"])  # estraiamo il PMID come identificatore del documento
            tokens = preprocess_and_tokenize(row.get("AbstractText", ""))  # tokenizza l'abstract

            doc_ids.append(doc_id)
            raw_tokenized_docs.append(tokens)

            # Document frequency (quanti documenti contengono il token almeno una volta)
            for unique_tok in set(tokens):
                df_counter[unique_tok] += 1

            if (idx + 1) % 50000 == 0:
                print(f"Indicizzati {idx + 1} documenti in {time.time() - start:.1f}s...")

    print(f"Completato ... Documenti: {len(doc_ids)}")
    # Applica min_df=10 filtrando i termini troppo rari
    valid_terms = {term for term, df in df_counter.items() if df >= min_df}
    filtered_corpus = [
        [tok for tok in doc_tokens if tok in valid_terms]
        for doc_tokens in raw_tokenized_docs
    ]

    bm25 = BM25Okapi(filtered_corpus)
    elapsed = time.time() - start

    index_artifact = {
        "index_name": "pubmed-index",
        "model": "bm25",
        "min_df": min_df,
        "doc_ids": doc_ids,
        "tokenized_corpus": filtered_corpus,
        "valid_terms": valid_terms,
        "bm25": bm25,
        "elapsed_seconds": elapsed,
    }

    with open(index_output_path, "wb") as file:
        pickle.dump(index_artifact, file)

    print(f"Completato in {elapsed:.2f}s | Documenti: {len(doc_ids)} | Termini (df>={min_df}): {len(valid_terms)}")
    print(f"Indice BM25 salvato in {index_output_path}\n")

    return index_artifact

def load_bm25_index() -> dict:
    index_path = DATA_PATH / "PubMed_BM25Okapi.pkl"

    if not index_path.exists():
        print(f"Indice BM25 non trovato in {index_path}. Creazione in corso...")
        return build_bm25_index(corpus_path, MIN_DF)

    with open(index_path, "rb") as file:
        index_artifact = pickle.load(file)

    print(f"Indice BM25 caricato da {index_path}")
    return index_artifact


# ------------------------------------------------- SEARCH BM25 -------------------------------------------------
def seach_bm25(results_path, inverted_index, query_list, top_k=20,):

    bm25 = inverted_index["bm25"]
    doc_ids = inverted_index["doc_ids"]

    results = {}
    for query in tqdm(query_list, desc="Ricerca BM25", unit="query"):
        idx = query["id"]
        query = query["text"]


        query = query or ""
        scores = bm25.get_scores(preprocess_and_tokenize(query))
        top_positions = np.argsort(scores)[::-1][:top_k]

        results[str(idx)] = {
            str(doc_ids[position]): round(float(scores[position]), 2)
            for position in top_positions
        }
    path = save_bm25_results_json(results, results_path)
    print(f"Risultati BM25 salvati in {path}")

    return results


def load_or_create_bm25_results(results_path) -> dict | None:
    """Controlla se esiste 'results_path' e lo carica come {idx: {pmid: score}}."""
    file_path = DATA_PATH / results_path

    if not file_path.exists():
        print(f"Nessun file '{results_path}' trovato su disco.\nPROCEDO A CREARE L'INDICE BM25 E A SALVARLO SU DISCO...")
        return None

    with open(file_path, mode="r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Normalizza le chiavi: idx -> int, pmid -> str, score -> float
    formatted_results = {
        int(idx): {str(pmid): float(score) for pmid, score in pmid_scores.items()}
        for idx, pmid_scores in raw_data.items()
    }

    print(f"Caricati {len(formatted_results)} risultati BM25 da '{results_path}'")
    return formatted_results



def save_bm25_results_json(results, results_path) -> Path:
    """Salva i risultati BM25 in formato JSON con nome 'results_path.json'."""
    file_path = DATA_PATH / results_path

    serializable_results = {
        str(idx): {str(pmid): float(score) for pmid, score in pmid_scores.items()}
        for idx, pmid_scores in results.items()
    }

    with open(file_path, mode="w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    return file_path
