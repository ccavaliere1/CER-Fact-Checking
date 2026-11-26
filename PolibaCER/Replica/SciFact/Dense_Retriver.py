import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from torch.utils.data import DataLoader
import logging

# Imposta logging per monitorare il processo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# File paths
corpus_path = "/content/drive/MyDrive/Scifact_elaborazione/corpus_train.csv"
questions_path = "/content/drive/MyDrive/Scifact_elaborazione/claims_train.csv"
output_path = "retrieval_results_dense_scifact.csv"

# Caricamento dei dati
logging.info("Caricamento dei file CSV...")
corpus = pd.read_csv(corpus_path)
queries = pd.read_csv(questions_path)

# Assicurati che le colonne esistano
assert 'abstract' in corpus.columns, "Il corpus deve avere una colonna 'abstract'."
assert 'claim' in queries.columns, "Il file delle query deve avere una colonna 'claim'."

# Prepara i documenti e le query
documents = corpus['abstract'].dropna().tolist()
claims = queries['claim'].dropna().tolist()

# Controllo dei documenti
if not documents:
    raise ValueError("La lista dei documenti è vuota.")
if not claims:
    raise ValueError("La lista delle query è vuota.")

# Verifica della GPU
if not torch.cuda.is_available():
    logging.error("GPU non disponibile. Assicurati di avere una GPU configurata correttamente.")
    raise EnvironmentError("GPU non disponibile.")
device = 'cuda'

# Caricamento del modello SBERT
logging.info("Caricamento del modello SBERT...")
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=device)

# Funzione per codificare i documenti in batch
def encode_in_batches(model, documents, batch_size=32):
    logging.info("Codifica dei documenti in batch...")
    dataloader = DataLoader(documents, batch_size=batch_size)
    embeddings = []
    for batch in dataloader:
        batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

# Codifica dei documenti
logging.info("Codifica dei documenti...")
document_embeddings = encode_in_batches(model, documents)

# Lista per i risultati finali
results = []

# Iterazione sulle query
logging.info("Avvio del retrieval denso...")
for idx, query in enumerate(claims):
    logging.info(f"Processamento query {idx + 1}/{len(claims)}: {query}")

    # Codifica della query
    query_embedding = model.encode(query, convert_to_tensor=True, device=device)

    # Calcolo dei punteggi di similarità
    dense_scores = util.cos_sim(query_embedding, document_embeddings).squeeze().tolist()

    # Ordina i documenti in base ai punteggi
    ranked_indices = np.argsort(dense_scores)[::-1]
    ranked_docs = [(documents[i], dense_scores[i]) for i in ranked_indices]

    # Salva i top 5 risultati per la query
    for rank, (doc, score) in enumerate(ranked_docs[:5], start=1):
        results.append({
            'query': query,
            'document': doc,
            'score': score,
            'rank': rank
        })

# Salva i risultati in un file CSV
logging.info("Salvataggio dei risultati in un file CSV...")
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

logging.info(f"Processo completato. Risultati salvati in: {output_path}")
