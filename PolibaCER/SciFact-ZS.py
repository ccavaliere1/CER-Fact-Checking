from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from transformers import pipeline
from PolibaCER.DataLoader.CorpusLoader import build_scifact_index
from PolibaCER.FactCheck.Retriver import retrieve_scifact_top_abstracts
from PolibaCER.FactCheck.Reasoning import reason_model_response
from PolibaCER.FactCheck.Justifyer import parse_response, generate_justification
from PolibaCER.FactCheck.Response import extract_label_and_score
from openai import OpenAI
import json


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

# -------------------------------------------- DATA SETUP ------------------------------------------- #
def get_zeroshoot_claim_and_labels(data_path):
    CLAIMS_DEV_PATH = data_path / "claims_dev.jsonl"
    # CLAIMS_DEV_PATH = data_path / "claims_train_snap.jsonl"

    claims = []
    labels = []

    with CLAIMS_DEV_PATH.open("r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
        for item in data:
            claims += [item['claim']]

            if item['evidence'] not in [None, {}]:# "evidence": {"13734012": [{"sentences": [4], "label": "CONTRADICT"}]}
                for doc_id, evidence_list in item['evidence'].items():
                    for evidence in evidence_list:
                        labels += [evidence['label']]
            else:
                labels += ["NEI"]


    return claims, labels

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
        model=MODEL,
        data_path=DATA_PATH,
        NEW_INDEX=False,
    )



    # ------------------------------------------- CLAIM INPUT ------------------------------------------- #
    claims, labels = get_zeroshoot_claim_and_labels(DATA_PATH)

    results={}

    for claim in claims:

        # ------------------------------------------- RETRIEVE TOP ABSTRACTS ------------------------------------------- #
        top_abstracts = retrieve_scifact_top_abstracts(
            claim=claim,
            model=MODEL,
            index=index,
            doc_ids=doc_ids,
            documents=documents,
            top_k=5
        )

        # IMPOSTAZIONE VALORI GLOBALI PER GLI ABSTRACTS
        # print("\n------------------------------------------- TOP ABSTRACTS -------------------------------------------")
        for i, result in enumerate(top_abstracts, start=1):
            globals()[f"abstract_{i}"] = result['abstract']
            globals()[f"docs_id{i}"] = result['doc_id']
            globals()[f"distance_{i}"] = result['distance']


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

        # ----------------------------------------- CREATE FINAL RESPONSE ----------------------------------------- #
        predicted_label, score_label = extract_label_and_score(result)

        # Calculate the confidence score
        confidence = f"{score_label * 100:.2f}%"

        results[claim] = {
            "predicted_label": predicted_label,
            "true_label": labels[claims.index(claim)],
            "confidence": confidence,
        }

    # for claim, result in results.items():
    #     print(f"Claim: {claim}")
    #     print(f"Predicted Label: {predicted_label} | Confidence: {confidence} ---> Label from dataset: {labels[claims.index(claim)]}")
    #     print("-" * 20)


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