from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from PolibaCER.DataLoader.DataLoader import build_scifact_index, save_prediction_to_json
from PolibaCER.FactCheck.Retriver import retrieve_scifact_top_abstracts
from PolibaCER.FactCheck.Reasoning import reason_model_response_LOCAL, reason_model_response_GEMINI
from PolibaCER.FactCheck.Justifyer import parse_response
from PolibaCER.FactCheck.Response import normalize_abstract_sentences
from openai import OpenAI
import json
from PolibaCER.SciFactTraning.traning_model import load_finetuned_model, predict
import os


# -------------------------------------------- GLOBAL VARIABLES ------------------------------------------- #
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data" / "SciFact"
RETRIVER_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
model_dir = BASE_PATH / "models" / "cer-deberta-v3-large-scifact"
TRAINED_CLASSIFIER_MODEL, tokenizer, device = load_finetuned_model(model_dir)
result_path = (Path(__file__).resolve().parents[1] / "DataLoader" / "TR_GEM3-5_CLJ.json")
CLAIMS_NUMBER = 100
LOCAL = False
NVIDIA = False


# -------------------------------------------- LLM RETRIVER_MODEL SETUP ------------------------------------------- #
if LOCAL:
    LLM_MODEL = "bartowski/Qwen2.5-7B-Instruct-GGUF:Q6_K"

    client = OpenAI(
        base_url="http://127.0.0.1:9931/v1",
        api_key="local"
    )
else:
    if NVIDIA:
        LLM_MODEL = "google/diffusiongemma-26b-a4b-it"  # funziona ma senza giustificazioni

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key= os.environ.get("NVIDIA_API_KEY")
        )
    else:
        # LLM_MODEL = "gemma-4-26b-a4b-it"
        # LLM_MODEL = "gemma-4-31b-it"
        LLM_MODEL = "gemini-3.5-flash-lite"
        #LLM_MODEL = "gemini-4-hydrogen"
        print("REMOTE LLM: GOOGLE GEMINI API - {} model".format(LLM_MODEL))

# -------------------------------------------- DATA SETUP ------------------------------------------- #

def get_claim_records(data_path, limit, random_selection=False):
    claims_path = data_path / "claims_test.jsonl"

    with claims_path.open("r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    data = data[:limit]

    records = []

    for item in data:
        evidence = item.get("evidence") or {}

        if not evidence:
            label = "NEI"
        else:
            labels = {
                annotation["label"].upper()
                for annotations in evidence.values()
                for annotation in annotations
            }

            if len(labels) != 1:
                raise ValueError(
                    f"Label multiple per claim {item['id']}: {labels}"
                )

            label = labels.pop()

        records.append({
            "id": item["id"],
            "claim": item["claim"],
            "true_label": label,
        })

    return records

def calculate_metrics(results):
    labels_order = ["SUPPORT", "CONTRADICT", "NEI"]

    label_mapping = {
        "TRUE": "SUPPORT",
        "SUPPORTED": "SUPPORT",
        "SUPPORT": "SUPPORT",
        "FALSE": "CONTRADICT",
        "CONTRADICTED": "CONTRADICT",
        "CONTRADICT": "CONTRADICT",
        "NEI": "NEI",
        "NOT ENOUGH INFORMATION": "NEI",
    }

    y_true = []
    y_pred = []

    for item in results.values():
        true_label = str(item["true_label"]).strip().upper()
        predicted_label = str(item["predicted_label"]).strip().upper()

        if true_label not in label_mapping:
            raise ValueError(f"True label non riconosciuta: {true_label}")

        if predicted_label not in label_mapping:
            raise ValueError(f"Predicted label non riconosciuta: {predicted_label}")

        y_true.append(label_mapping[true_label])
        y_pred.append(label_mapping[predicted_label])

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_order,
        target_names=labels_order,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    return report


# -------------------------------------------- MAIN FUNCTION ------------------------------------------- #
if __name__ == "__main__":

    # ------------------------------------------- BUILD SCI-FACT INDEX ------------------------------------------- #
    documents, doc_ids, index = build_scifact_index(
        model=RETRIVER_MODEL,
        data_path=DATA_PATH,
        NEW_INDEX=False,
    )



    # ------------------------------------------- CLAIM INPUT ------------------------------------------- #
    records = get_claim_records(DATA_PATH, CLAIMS_NUMBER)

    results={}

    for record_index, record in enumerate(records):

        # ------------------------------------------- RETRIEVE TOP ABSTRACTS ------------------------------------------- #
        top_abstracts = retrieve_scifact_top_abstracts(
            claim=record["claim"],
            model=RETRIVER_MODEL,
            index=index,
            doc_ids=doc_ids,
            documents=documents,
            top_k=4
        )

        # IMPOSTAZIONE VALORI GLOBALI PER GLI ABSTRACTS
        # print("\n------------------------------------------- TOP ABSTRACTS -------------------------------------------")
        for i, result in enumerate(top_abstracts, start=1):
            globals()[f"abstract_{i}"] = result['abstract']
            globals()[f"docs_id{i}"] = result['doc_id']
            globals()[f"distance_{i}"] = result['distance']


        # ------------------------------------------- PROCESSING OF THE TOP ABSTRACTS -------------------------------------------#
        if NVIDIA or LOCAL:
            answer = reason_model_response_LOCAL(globals(), record["claim"], top_abstracts, client, LLM_MODEL)
        else:
            answer = reason_model_response_GEMINI(globals(), record["claim"], top_abstracts, LLM_MODEL)

        # ------------------------------------------- JUSTIFICATION GENERATION ------------------------------------------- #

        first_label, justification, supporting, refusing, notes = parse_response(answer)

        # LA FUNZIONE PREDICT EQUIVALE ALLA "generate_justification" del ZEROSHOOT RETRIVER_MODEL
        # Generate the justification text
        justification_text = (
            f'Justification: "{justification}"'
        )

        # Limit the justification text to a maximum length
        max_length = 512
        if len(justification_text) > max_length:
            justification_text = justification_text[:max_length]
            print(f"Justification text truncated to {max_length} characters.")

        result = predict(
            model=TRAINED_CLASSIFIER_MODEL,
            tokenizer=tokenizer,
            claim=record["claim"],
            justification=justification_text,
            device=device,
        )

        # ----------------------------------------- CREATE FINAL RESPONSE ----------------------------------------- #
        probabilities = result["probabilities"]

        predicted_label, confidence = max(
            probabilities.items(),
            key=lambda item: item[1],
        )

        if not LOCAL:  # BISOGNA EFFETTUARE UN ULTERIORE PULIZIA DEI SUPPORTING E REFUSING PERCHÉ IL MODELLO GEMMA NON RESTITUISCE SEMPRE LE FRASI IN MODO PULITO
            supporting = normalize_abstract_sentences(supporting)
            refusing = normalize_abstract_sentences(refusing)


        results[record["claim"]] = {
            "predicted_label": predicted_label,
            "true_label": record["true_label"],
            "confidence": confidence,
        }

        # ----------------------------------------- SAVE PREDICTION TO JSON ----------------------------------------- #
        save_prediction_to_json(
            id=record_index,
            predicted_label=predicted_label,
            true_label=record["true_label"],
            output_file=result_path
        )

    for claim, prediction in results.items():
        print(f"Claim: {claim}")
        print(f"Predicted Label: {prediction['predicted_label']} Confidence: {prediction['confidence']} ---> Real Label: {prediction['true_label']}")
        print("-" * 20)


    # ------------------------------------------- CONFUSION MATRIX ------------------------------------------- #

    metrics = calculate_metrics(results)

    print("\nMetriche per classe:")
    for label in ["SUPPORT", "CONTRADICT", "NEI"]:
        print(
            f"{label}: "
            f"precision={metrics[label]['precision']:.4f}, "
            f"recall={metrics[label]['recall']:.4f}, "
            f"F1={metrics[label]['f1-score']:.4f}"
        )

    print("\nMetriche aggregate:")
    print(f"Macro precision: {metrics['macro avg']['precision']:.4f}")
    print(f"Macro recall:    {metrics['macro avg']['recall']:.4f}")
    print(f"Macro F1:        {metrics['macro avg']['f1-score']:.4f}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")