from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PolibaCER.DataLoader.CorpusLoader import load_pubmed_embeddings
from PolibaCER.FactCheck.Retriver import retrieve_pubmed_top_abstracts
from PolibaCER.FactCheck.Reasoning import reason_model_response
from PolibaCER.FactCheck.Justifyer import parse_response, generate_justification
from PolibaCER.FactCheck.Response import extract_label_and_score, extract_supporting_and_refusing_sentences, proccess_abstracts
from PolibaCER.FactCheck.ClaimDetection import input_claim_text
from openai import OpenAI


# -------------------------------------------- GLOBAL VARIABLES ------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"

embeddings_file = str(DATA_PATH) + "\\abstract_embeddings.npy"
pmid_file = str(DATA_PATH) + "\\pmids.npy"
faiss_index_file = str(DATA_PATH) + "\\faiss_index.index"
file_path = str(DATA_PATH) + "\\parte_205.csv"

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
ZEROSHOT_CLASSIFIER_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
LOCAL = True


# -------------------------------------------- LLM MODEL SETUP ------------------------------------------- #
if LOCAL:
    LLM_MODEL = "bartowski/Qwen2.5-7B-Instruct-GGUF:Q6_K"

    client = OpenAI(
        base_url="http://127.0.0.1:9931/v1",
        api_key="local"
    )
else:
    LLM_MODEL = "google/diffusiongemma-26b-a4b-it"  # funziona ma senza giustificazioni

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=st.secrets.nvidia.api_key
    )


# -------------------------------------------- MAIN FUNCTION ------------------------------------------- #
if __name__ == "__main__":

    # ------------------------------------------- LOAD DATA ------------------------------------------- #
    # Load data
    data = pd.read_csv(file_path)
    embeddings, pmids, index = load_pubmed_embeddings(embeddings_file, pmid_file, faiss_index_file)


    # ------------------------------------------- CLAIM INPUT ------------------------------------------- #
    claim = input_claim_text("Vitamin D reduces the risk of osteoporosis.")


    # ------------------------------------------- RETRIEVE TOP ABSTRACTS ------------------------------------------- #
    top_abstracts = retrieve_pubmed_top_abstracts(claim=claim, model=MODEL, index=index, pmids=pmids, data=data, top_k=5)

    # IMPOSTAZIONE VALORI GLOBALI PER GLI ABSTRACTS
    print("\n------------------------------------------- TOP ABSTRACTS -------------------------------------------")
    for i, (abstract, pmid, distance) in enumerate(top_abstracts, 1):
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        globals()[f"abstract_{i}"] = abstract
        globals()[f"reference_{i}"] = pubmed_url
        globals()[f"distance_{i}"] = distance

        # PRINT RESULTS
        print(f"\n[{i}] doc_id={pmid}")
        print(f"Abstract: {abstract}")
        print(f"Reference: {pubmed_url}")
        print(f"Distance: {distance:.4f}")
    print("---------------------------------------------------------------------------------------------------")


    # ------------------------------------------- PROCESSING OF THE TOP ABSTRACTS -------------------------------------------#
    answer = reason_model_response(globals(), claim, top_abstracts, client, LLM_MODEL)


    # ------------------------------------------- JUSTIFICATION GENERATION ------------------------------------------- #
    zeroshot_classifier = pipeline(
        "zero-shot-classification", model=ZEROSHOT_CLASSIFIER_MODEL
    )
    first_label, justification, supporting, refusing, notes = parse_response(answer)

    # Generate the justification for the claim
    # in realtà classifica la giustificazione con il claim, la justification é generata dal modello di ragionamento
    result = generate_justification(zeroshot_classifier, claim, justification)

    # Justification Results
    print("\n------------------------------------------- JUSTIFICATION RESULTS -------------------------------------------")
    print(f"Label: {first_label}")
    print(f"Justification: {justification}")
    print(f"Supporting sentences from abstracts: {supporting}")
    print(f"Refusing sentences from abstracts: {refusing}")
    print(f"Note: {notes}")
    print("-------------------------------------------------------------------------------------------------------------")


    # ----------------------------------------- CREATE FINAL RESPONSE ----------------------------------------- #
    predicted_label, score_label = extract_label_and_score(result)

    # Calculate the confidence score
    confidence = f"{score_label * 100:.2f}%"

    abstracts, supporting, refusing = extract_supporting_and_refusing_sentences(globals(), top_abstracts, supporting, refusing)

    processed_abstracts, refusing_matches = proccess_abstracts(abstracts, supporting, refusing)

    print("\n------------------------------------------- FINAL RESULTS -------------------------------------------")
    print(f"Claim: {claim}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence Score: {confidence}")
    print("\nProcessed Abstracts:")
    if not processed_abstracts:
        print("No processed abstracts found.")
    else:
        for abstract_name, abstract_text in processed_abstracts.items():
            print(f"\n{abstract_name}: {abstract_text}")
    if not refusing_matches:
        print("\nNo refusing matches found.")
    else:
        print("\nRefusing Matches:")
        for match in refusing_matches:
            print(f"{match['abstract']}: {match['text']}")

    print("---------------------------------------------------------------------------------------------------")
