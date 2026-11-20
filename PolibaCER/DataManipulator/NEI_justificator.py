from pathlib import Path
from huggingface_hub.utils import tqdm
from sentence_transformers import SentenceTransformer
from PolibaCER.FactCheck.GoogleAPI import gemini_generate
from openai import OpenAI
import json
import os


# -------------------------------------------- GLOBAL VARIABLES ------------------------------------------- #
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data" / "SciFact"
RETRIVER_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CLAIMS_NUMBER = 304
LOCAL = True
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

def get_claim_records(claims_path, limit):

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

def get_abstracts_for_claims(claims_path, limit):
    corpus_path = DATA_PATH / "corpus.jsonl"

    with claims_path.open("r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    with corpus_path.open("r", encoding="utf-8") as file:
        corpus = [json.loads(line) for line in file]

    data = data[:limit]

    abstracts = {}

    for item in data:
        cited_doc_ids = item.get("cited_doc_ids") or {}
        abstracts[item["id"]] = []

        if not cited_doc_ids:
            abstracts[item["id"]].append("")
        else:
            for doc_id in cited_doc_ids:
                doc = next((d for d in corpus if d["doc_id"] == doc_id), None)
                if doc:
                    abstracts[item["id"]].append(doc.get("abstract", ""))
                else:
                    abstracts[item["id"]].append("")

    return abstracts

# -------------------------------------------- SALVARE LA GIUSTIFICAZIONE ------------------------------------------- #
def save_justifications(input_path: Path, output_path: Path, justifications: dict[int, str]) -> None:
    """
    Salva una justification nel record corrispondente all'id SciFact.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    updated_ids = set()

    for record in records:
        record_id = int(record["id"])

        if record_id in justifications:
            justification = justifications[record_id].strip()

            if not justification:
                raise ValueError(
                    f"Justification vuota per il claim {record_id}."
                )

            record["justification"] = justification
            updated_ids.add(record_id)

    missing_ids = set(justifications) - updated_ids

    if missing_ids:
        raise KeyError(
            f"Claim non presenti nel dataset: {sorted(missing_ids)}"
        )

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def llm_reasoning_template(query):
    """
    Generate a template for the prompt used for justification generation by the LLM model.

    Args:
        query (str): Claim to fact-check.

    Returns:
        str: Reasoning template for the claim.
    """
    llm_nei_factual_prompt = """<<SYS>> [INST]

        You are a precise biomedical evidence extraction assistant.
        Given a claim and scientific abstracts (where the topic or claim may be cited or mentioned), your task is to extract verbatim sentences from the abstracts that are most relevant to the claim or show the scope/limits of what the abstracts actually report (helping explain why the claim cannot be confirmed or refuted).
        
        Strict Rules:
        - Copy sentences STRICTLY VERBATIM (character-for-character / word-for-word) from the provided abstracts.
        - DO NOT paraphrase, rewrite, summarize, translate, complete, or alter punctuation or wording in any way.
        - DO NOT invent sentences that are not explicitly present in the abstracts.
        - Only include sentences that actually appear in the text. If no sentence in an abstract is relevant or useful, write the absence of relevant information.
        - Do not add explanations outside the requested sections.
        - Keep the output format strictly as specified below (max 2 sentences)
        
        Output format:
        <exact verbatim sentence>
        
        (or write the absence of relevant information if no suitable sentence exists)
        
        [/INST] <</SYS>>
        
        [INST] Claim: {claim} [/INST]
        [INST] Abstracts:
        """
    return llm_nei_factual_prompt



# -------------------------------------------- MAIN FUNCTION ------------------------------------------- #
if __name__ == "__main__":

    claims_path = DATA_PATH / "claims_train_nei.jsonl"

    # ------------------------------------------- CLAIM INPUT ------------------------------------------- #
    records = get_claim_records(claims_path, CLAIMS_NUMBER)

    results={}
    justifications = {}

    for record_index, record in tqdm(enumerate(records), total=len(records), desc="Processing claims"):

        # ------------------------------------------- ESTRAZIONE ABSTRACTS ------------------------------------------- #


        abstracts = get_abstracts_for_claims(claims_path, CLAIMS_NUMBER)

        # ------------------------------------------- PROCESSING OF THE TOP ABSTRACTS -------------------------------------------#
        if NVIDIA or LOCAL:
            prompt_template = llm_reasoning_template(record["claim"])

            for i, abstract in enumerate(abstracts[record["id"]], start=1):
                prompt_template += f"{abstract} ; "
            prompt_template += "[/INST]"

            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt_template}],
                temperature=0.1,
                top_p=0.7,
                max_tokens=1024,
                stream=True
            )

            # Collect the response
            answer = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content

            # print(prompt_template)
            # print(f"Claim ID: {record['id']}, Justification: {answer}")
        else:
            prompt_template = llm_reasoning_template(record["claim"])

            # Add the abstracts to the prompt
            for i in range(1, len(top_abstracts)):
                prompt_template += f"{globals()[f'abstract_{i}']} ; "
            prompt_template += f"{globals()[f'abstract_{i + 1}']} [/INST]"

            answer = gemini_generate(prompt_template, model=LLM_MODEL)

        # ------------------------------------------- JUSTIFICATION GENERATION ------------------------------------------- #

        justifications[record["id"]] = answer

    # ------------------------------------------- SAVE JUSTIFICATIONS ------------------------------------------- #
    save_justifications(
        input_path=DATA_PATH / "claims_train_nei.jsonl",
        output_path=DATA_PATH / "claims_train_justifications_processed.jsonl",
        justifications=justifications,
    )