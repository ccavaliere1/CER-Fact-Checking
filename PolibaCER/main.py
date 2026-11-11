from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PolibaCER.DataLoader.CorpusLoader import build_scifact_index
from PolibaCER.FactCheck.Retriver import retrieve_top_abstracts
from PolibaCER.FactCheck.Reasoning import reason_model_response
from PolibaCER.FactCheck.Justifyer import parse_response, generate_justification
from PolibaCER.FactCheck.Response import extract_label_and_score, extract_supporting_and_refusing_sentences, proccess_abstracts
from ClaimDetection import input_claim_text
from openai import OpenAI


# -------------------------------------------- GLOBAL VARIABLES ------------------------------------------- #
DATA_PATH = Path(__file__).resolve().parent / "data" / "SciFact"
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

    # ------------------------------------------- BUILD SCI-FACT INDEX ------------------------------------------- #
    documents, doc_ids, index = build_scifact_index(
        model=MODEL,
        data_path=DATA_PATH,
        NEW_INDEX=False,
    )


    # ------------------------------------------- CLAIM INPUT ------------------------------------------- #
    claim = input_claim_text("Vitamin D reduces the risk of osteoporosis.")


    # ------------------------------------------- RETRIEVE TOP ABSTRACTS ------------------------------------------- #
    top_abstracts = retrieve_top_abstracts(
        claim=claim,
        model=MODEL,
        index=index,
        doc_ids=doc_ids,
        documents=documents,
        top_k=5,
    )

    # IMPOSTAZIONE VALORI GLOBALI PER GLI ABSTRACTS
    print("\n------------------------------------------- TOP ABSTRACTS -------------------------------------------")
    for i, result in enumerate(top_abstracts, start=1):
        globals()[f"abstract_{i}"] = result['abstract']
        globals()[f"docs_id{i}"] = result['doc_id']
        globals()[f"distance_{i}"] = result['distance']

        # PRINT RESULTS
        print(f"\n[{i}] doc_id={result['doc_id']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Titolo: {result['title']}")
        print(f"Abstract: {result['abstract']}")
    print("\n---------------------------------------------------------------------------------------------------\n")


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
    print(f"\n\nLabel: {first_label}")
    print(f"Justification: {justification}")
    print(f"Supporting sentences from abstracts: {supporting}")
    print(f"Refusing sentences from abstracts: {refusing}")
    print(f"Note: {notes}")
    print("\n-----------------------------------------------------------------------------------------------------------\n")


    # ----------------------------------------- CREATE FINAL RESPONSE ----------------------------------------- #
    predicted_label, score_label = extract_label_and_score(result)

    # Calculate the confidence score
    confidence = f"{score_label * 100:.2f}%"

    abstracts, supporting, refusing = extract_supporting_and_refusing_sentences(globals(), top_abstracts, supporting, refusing)

    processed_abstracts, refusing_matches = proccess_abstracts(abstracts, supporting, refusing)

    print("\n------------------------------------------- FINAL RESULTS -------------------------------------------")
    print(f"\nClaim: {claim}")
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

    print("\n---------------------------------------------------------------------------------------------------\n")
