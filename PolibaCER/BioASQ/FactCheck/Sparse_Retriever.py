import csv
import sys
from collections import defaultdict
from pathlib import Path
import torch
from huggingface_hub.utils import tqdm
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import nltk
from PolibaCER.BioASQ.DataCreator.DatasetMaker import get_bioasq_data
from PolibaCER.BioASQ.FactCheck.BM25 import seach_bm25, load_or_create_bm25_results, load_bm25_index
nltk_data_dir = Path(sys.prefix) / "nltk_data"
nltk.download("punkt_tab", download_dir=str(nltk_data_dir))
nltk.data.path.insert(0, str(nltk_data_dir))

# -------------------------------------------------- START UP DATA -------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data"
sentence_model = SentenceTransformer('copenlu/spiced')
start_idx = 0  # start
end_idx = 1048  # end


# -------------------------------------------------- LOAD BM25 INDEX AND BIOASQ DATA ---------------------------------------------
inverted_index = load_bm25_index()

bioasq_data = get_bioasq_data()
claims = bioasq_data["questions"]
labels = bioasq_data["answers"]


query_list = []
for idx, claim in enumerate(claims[start_idx:end_idx]):
    claim = claim.lower()
    query_list.append({"id": str(idx + start_idx), "text": claim})


results = load_or_create_bm25_results("BioASQ_BM25_results.json")
if results is None:
    results = seach_bm25("BioASQ_BM25_results.json", inverted_index, query_list, top_k=20)


# ------------------------------------------------- SELECT TOP SENTENCES -------------------------------------------------
# Carica il file CSV degli abstract
abstracts = pd.read_csv(DATA_PATH / "PubMed_p205.csv")

# Crea un dizionario di mapping dagli ID degli abstract ai loro testi
abstract_dict = pd.Series(abstracts.AbstractText.values, index=abstracts.PMID.astype(str)).to_dict()

# Rimappa tutti gli ID dei documenti dai risultati.
remapped_all_ids = []
for claim_id, document_results in tqdm(results.items(), desc="Remapping document IDs", unit="claim"):
    remapped_ids = []
    for doc_id, score in document_results.items():
        if doc_id in abstract_dict:
            remapped_ids.append(doc_id)
        else:
            print(f"Skipping invalid doc_id: {doc_id}")
    remapped_all_ids.append(remapped_ids)

# Crea una lista di frasi per ciascun documento associato a ciascun claim.
claim_sentences = []
for ids in tqdm(remapped_all_ids, desc="Tokenizing abstracts", unit="abstract"):
    all_sentences = []
    for doc_id in ids:
        abstract = abstract_dict[doc_id]
        sentences = sent_tokenize(abstract)
        all_sentences.append((doc_id, sentences))
    claim_sentences.append(all_sentences)

# Trova le prime 3 frasi più simili alla query da ciascun documento, per condensarlo nelle parti più importanti.
top_sentences = []
for idx in tqdm(range(start_idx, min(end_idx, len(claims))), desc="Processing claims", unit="claim"):
    claim = claims[idx]

    if idx - start_idx >= len(claim_sentences):
        print(f"Warning: claim index {idx} out of range")
        continue

    sents_per_doc = claim_sentences[idx - start_idx]

    doc_top_sentences = []
    for doc_id, sents in sents_per_doc:
        sents_embeddings = sentence_model.encode(sents, convert_to_tensor=True)
        claim_embedding = sentence_model.encode(claim, convert_to_tensor=True)
        cos_scores = util.cos_sim(claim_embedding, sents_embeddings)[0]

        new_k = 3
        if len(sents) < 3:
            new_k = len(sents)

        top_results = torch.topk(cos_scores, k=new_k)

        np_results = top_results[1].detach().cpu().numpy()
        np_results = np_results[np_results < len(sents)]  # Filtra gli indici fuori range
        doc_top_sentences.append((doc_id, np_results))
    top_sentences.append(doc_top_sentences)

# Crea una lista di frasi selezionate per ciascun claim, basata sulle frasi più simili.
selected_sentences = []
for doc_sents, doc_top_sents in tqdm(zip(claim_sentences, top_sentences), desc="Selecting top sentences", unit="claim"):
    doc_selected_sents = []
    for (doc_id, sents), (top_doc_id, top) in zip(doc_sents, doc_top_sents):
        selected_sents = np.array(sents)[top]
        doc_selected_sents.append((doc_id, selected_sents))
    selected_sentences.append(doc_selected_sents)

# Crea una lista congiunta di claims concatenati e prove, in forma di "claim [SEP] evidence1 evidence2 evidence3"
joint_list = []
for idx in tqdm(range(start_idx, min(end_idx, len(claims))), desc="Creating joint list", unit="claim"):
    claim = claims[idx]
    for doc_id, sents in selected_sentences[idx - start_idx]:
        joint = claim + " [SEP] "
        for s in sents:
            joint += s + f" [{doc_id}] "
        joint_list.append(joint)

# ---------------------------------------------------- SAVE TO FILE TO TXT ---------------------------------------------
# Salva la lista congiunta in un file di testo
output_filename = DATA_PATH / "BioASQ_k20_sentences.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    for line in joint_list:
        f.write(line)
        f.write("\n")

print(f"File CSV generato: {output_filename}")

# --------------------------------------------------- SAVE TO FILE TO CSV ---------------------------------------------
# Salva la lista congiunta in un file CSV
# Leggere il file di testo output_filename che é il risultato salvato nel file txt
with open(output_filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Dizionario per raggruppare le frasi e le relative sentence
frase_dict = defaultdict(list)

# Parsing delle linee
for line in lines:
    if '[SEP]' in line:
        # Separare la frase e la sentence
        frase, sentence = line.split('[SEP]')
        sentence = sentence.strip()

        # Aggiungere la sentence al dizionario
        frase_dict[frase.strip()].append(sentence)

# Scrittura nel file CSV
csv_file_path = DATA_PATH / 'BioASQ_sentences_true_label.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Scrivere l'intestazione
    csvwriter.writerow(['Frase', 'Sentences Concatenate'])

    # Scrivere i dati raggruppati
    for frase, sentences in tqdm(frase_dict.items(), desc="Writing to CSV", unit="frase"):
        sentences_concat = ' '.join(sentences)
        csvwriter.writerow([frase, sentences_concat])

print(f"File CSV generato: {csv_file_path}")