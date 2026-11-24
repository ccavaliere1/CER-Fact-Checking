import pickle
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import nltk


nltk.download('punkt')


start_idx = 0  # start
end_idx = 1048  # end


with open('/content/drive/MyDrive/pickled_sr', 'rb') as file:
    inverted_index = pickle.load(file)


df = pd.read_csv("/content/drive/MyDrive/healthFC_annotated.csv")
claims = df.en_claim.tolist()
labels = df.label.tolist()


query_list = []
for idx, claim in enumerate(claims[start_idx:end_idx]):
    claim = claim.lower()
    query_list.append({"id": str(idx + start_idx), "text": claim})


results = inverted_index.msearch(
  queries=query_list,
  cutoff=20,
)

# Carica il file CSV degli abstract
abstracts = pd.read_csv("/content/drive/MyDrive/abstract_merged.csv")

# Crea un dizionario di mapping dagli ID degli abstract ai loro testi
abstract_dict = pd.Series(abstracts.AbstractText.values, index=abstracts.PMID.astype(str)).to_dict()

# Rimappa tutti gli ID dei documenti dai risultati.
remapped_all_ids = []
for claim_id, document_results in results.items():
    remapped_ids = []
    for doc_id, score in document_results.items():
        if doc_id in abstract_dict:
            remapped_ids.append(doc_id)
        else:
            print(f"Skipping invalid doc_id: {doc_id}")
    remapped_all_ids.append(remapped_ids)

claim_sentences = []
for ids in remapped_all_ids:
    all_sentences = []
    for doc_id in ids:
        abstract = abstract_dict[doc_id]
        sentences = sent_tokenize(abstract)
        all_sentences.append((doc_id, sentences))
    claim_sentences.append(all_sentences)

# Carica il modello di trasformatore di frasi per selezionare le frasi di prova.
model = SentenceTransformer('copenlu/spiced')
print("loaded sentence model!")

# Trova le prime 3 frasi più simili alla query da ciascun documento, per condensarlo nelle parti più importanti.
top_sentences = []
for idx in range(start_idx, min(end_idx, len(claims))):
    claim = claims[idx]
    print(f"Processing claim {idx + 1}/{len(claims)}")

    if idx - start_idx >= len(claim_sentences):
        print(f"Warning: claim index {idx} out of range")
        continue

    sents_per_doc = claim_sentences[idx - start_idx]

    doc_top_sentences = []
    for doc_id, sents in sents_per_doc:
        sents_embeddings = model.encode(sents, convert_to_tensor=True)
        claim_embedding = model.encode(claim, convert_to_tensor=True)
        cos_scores = util.cos_sim(claim_embedding, sents_embeddings)[0]

        new_k = 3
        if len(sents) < 3:
            new_k = len(sents)

        top_results = torch.topk(cos_scores, k=new_k)

        np_results = top_results[1].detach().cpu().numpy()
        np_results = np_results[np_results < len(sents)]  # Filtra gli indici fuori range
        doc_top_sentences.append((doc_id, np_results))
    top_sentences.append(doc_top_sentences)

selected_sentences = []
for doc_sents, doc_top_sents in zip(claim_sentences, top_sentences):
    doc_selected_sents = []
    for (doc_id, sents), (top_doc_id, top) in zip(doc_sents, doc_top_sents):
        selected_sents = np.array(sents)[top]
        doc_selected_sents.append((doc_id, selected_sents))
    selected_sentences.append(doc_selected_sents)

# Crea una lista congiunta di claims concatenati e prove, in forma di "claim [SEP] evidence1 evidence2 evidence3"
joint_list = []
for idx in range(start_idx, min(end_idx, len(claims))):
    claim = claims[idx]
    for doc_id, sents in selected_sentences[idx - start_idx]:
        joint = claim + " [SEP] "
        for s in sents:
            joint += s + f" [{doc_id}] "
        joint_list.append(joint)

output_filename = f"/content/drive/MyDrive/20_sentences_SciFact.txt"
with open(output_filename, "w") as f:
    for line in joint_list:
        f.write(line)
        f.write("\n")

with open(output_filename, "r") as f:
    lines = f.readlines()
    flatlines = []
    for l in lines:
        l = l.strip()
        if "[SEP]" in l:
            flatlines.append(l)

print("Processing complete!")