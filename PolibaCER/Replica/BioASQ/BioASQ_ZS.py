import re
from pathlib import Path
import pandas as pd
from huggingface_hub.utils import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from PolibaCER.Replica.FactCheck.Justifiyer import justify_dataframe
from PolibaCER.Replica.FactCheck.Sparse_Retriever import start_BM25_on_BioASQ
from PolibaCER.Replica.BioASQ.DataCreator.DatasetMaker import get_bioasq_data
from PolibaCER.Replica.FactCheck.BM25 import load_bm25_index
from PolibaCER.Replica.FactCheck.ZS_DeBerta import generate_justification

DATA_PATH = Path(__file__).resolve().parents[0] / "data"
bioASQ_path = DATA_PATH / "BioASQ_k20_sentences.csv"
final_result_name = "BioASQ_ZS_final_results"
LLM_MODEL = "bartowski/Qwen2.5-7B-Instruct-GGUF:Q6_K"
START_INDEX = 0
NUM_ROWS = 744

# -------------------------------------------- LOAD DATA ------------------------------------------- #
bioasq_data = get_bioasq_data()
claims = bioasq_data["questions"]
true_labels = bioasq_data["answers"]

# -------------------------------------------- LOAD BM25 INDEX ------------------------------------------- #
inverted_index = load_bm25_index()

# -------------------------------------------- START BM25 RETRIVAL ------------------------------------------- #
if not bioASQ_path.exists():
    print(f"Nessun file '{bioASQ_path}' trovato su disco.\nPROCEDO A CREARE L'INDICE BM25 E A SALVARLO SU DISCO...")
    start_BM25_on_BioASQ(inverted_index, claims, output_filename="BioASQ_k20_sentences")

# --------------------------------------------- JUSTIFY DATAFRAME ------------------------------------------- #
dataframe = pd.read_csv(bioASQ_path)
print("Dataframe loaded successfully. Starting justification process...")
dfj = justify_dataframe(dataframe, start_index=START_INDEX, num_rows=NUM_ROWS, llm_model=LLM_MODEL)

# Save the justified dataframe to a CSV file
justificated_outpath = DATA_PATH / "BioASQ_justified_results.csv"
dfj.to_csv(justificated_outpath, index=False)
print(f"Justified dataframe saved to {justificated_outpath}")

# --------------------------------------------- FINAL DATAFRAME ------------------------------------------- #
justified_rows = dfj.loc[
    dfj["Justification"].notna()
    & dfj["Justification"].astype(str).str.strip().ne("")
].copy()

true_labels_dataframe = pd.DataFrame({
    "Frase": claims,
    "True Label": true_labels,
})

source_dataframe = dataframe.merge(true_labels_dataframe, on="Frase", how="left", validate="one_to_one")
final_dataframe = source_dataframe.merge(justified_rows, left_on="Frase", right_on="Claim", how="inner", validate="one_to_one")

if final_dataframe["True Label"].isna().any():
    print("Count of missing 'True Label' values:", final_dataframe["True Label"].isna().sum())
    final_dataframe["True Label"] = final_dataframe["True Label"].fillna("NEI")
    print("Warning: Filled dataframe with 'NEI' for missing 'True Label' values.")

final_dataframe = final_dataframe.rename(columns={
    "Predicted label": "Predicted Label",
    "Sentences Concatenate": "Sentence_concatenate",
})[[
    "Claim",
    "True Label",
    "Predicted Label",
    "Sentence_concatenate",
    "Justification",
]]

final_dataframe.to_csv(DATA_PATH / f"{final_result_name}_no_classfied.csv", index=False)

# -------------------------------------------- ZERO-SHOT CLASIFICATION ------------------------------------------- #
results = []
with open(f"{DATA_PATH / final_result_name}.txt", 'w', encoding='utf-8') as file:
    for idx, (_, row) in tqdm(enumerate(final_dataframe.iterrows()), desc="Classifying claims", unit="claim", total=len(final_dataframe)):
        validity = generate_justification(row, classes=["yes", "no"])
        output_string = f"row {idx}: {validity}"
        results.append(output_string)
        file.write(output_string + "\n")

# SALVA I RISULTATI IN UN FILE CSV
def parse_line(line):
    sequence_match = re.search(r"'sequence': '(.*?)'", line)
    label_match = re.search(r"'labels': \['(.*?)', '(.*?)'\]", line)

    sequence = sequence_match.group(1) if sequence_match else ''
    label = label_match.group(1) if label_match else ''

    return sequence, label #, true_label

data = []
with open(f"{DATA_PATH / final_result_name}.txt", 'r', encoding='utf-8') as infile:
    for line in infile:
        sequence, label = parse_line(line)
        true_label = final_dataframe.loc[final_dataframe['Claim'] == sequence, 'True Label'].values[0] if not final_dataframe.loc[final_dataframe['Claim'] == sequence].empty else 'NEI'
        data.append([sequence, label, true_label])
print(f"Risultati zero-shot salvati in {str(DATA_PATH)}/{final_result_name}.csv")

# Crea un DataFrame
df = pd.DataFrame(data, columns=['sequence', 'label', 'True Label'])
df.to_csv(DATA_PATH / f"{final_result_name}.csv", index=False)

# -------------------------------------------- METRICS ------------------------------------------- #
precision = precision_score(final_dataframe["True Label"], df["label"], average="macro")
recall = recall_score(final_dataframe["True Label"], df["label"], average="macro")
f1 = f1_score(final_dataframe["True Label"], df["label"], average="macro")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")