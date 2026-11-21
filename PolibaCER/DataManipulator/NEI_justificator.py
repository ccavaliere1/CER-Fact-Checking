from pathlib import Path
from typing import Any
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
CLAIMS_NUMBER = 809
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
        LLM_MODEL = "mistralai/mistral-large-4-1t-a52b"

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
        data = [json.loads(line) for line in file][:limit]

    with corpus_path.open("r", encoding="utf-8") as file:
        corpus = {
            str(document["doc_id"]): document
            for line in file
            if (document := json.loads(line))
        }

    abstracts = {}

    for item in data:
        evidence = item.get("evidence") or {}
        claim_abstracts = []

        labels = {
            annotation.get("label", "").upper()
            for annotations in evidence.values()
            for annotation in annotations
        }

        if labels - {"SUPPORT", "CONTRADICT"}:
            raise ValueError(
                f"Label non valida per il claim {item['id']}: {labels}"
            )

        if labels:
            # SUPPORT/CONTRADICT: solo le frasi indicate nelle evidence.
            claim_abstracts.append(evidence_to_text(evidence, corpus))
        else:
            # NEI: intero abstract dei documenti citati.
            for doc_id in item.get("cited_doc_ids") or []:
                document = corpus.get(str(doc_id))
                if document is None:
                    raise KeyError(
                        f"Documento {doc_id} non presente nel corpus SciFact."
                    )

                claim_abstracts.append(
                    " ".join(document.get("abstract") or [])
                )

        abstracts[item["id"]] = claim_abstracts

    return abstracts


def evidence_to_text(
    evidence: dict[str, list[dict[str, Any]]],
    corpus: dict[str, dict[str, Any]],
) -> str:
    sentences = []

    for doc_id, annotations in evidence.items():
        document = corpus.get(str(doc_id))

        if document is None:
            raise KeyError(
                f"Documento {doc_id} non presente nel corpus SciFact."
            )

        abstract = document.get("abstract") or []

        for annotation in annotations:
            for sentence_id in annotation.get("sentences", []):
                if not isinstance(sentence_id, int):
                    raise TypeError(
                        f"Indice frase non valido: {sentence_id}"
                    )

                if sentence_id < 0 or sentence_id >= len(abstract):
                    raise IndexError(
                        f"Frase {sentence_id} non valida "
                        f"per il documento {doc_id}."
                    )

                sentences.append(abstract[sentence_id])

    return " ".join(sentences)

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


def llm_reasoning_template(query, label=None):
    """
    Generate a template for the prompt used for justification generation by the LLM model.

    Args:
        query (str): Claim to fact-check.

    Returns:
        str: Reasoning template for the claim.
    """
    prompt_nei = f"""<<SYS>> [INST]

        You are a helpful, respectful and honest Doctor. Always answer as helpfully as possible using the context text provided.

        Use the information in Context.
    
        Elaborate the Context to generate a new information.
    
        Use only the knowledge in Context to answer.
        
        Be strict about using only the information provided in the Context from the scientific papers.
    
        Answer describing in a scientific way. Be formal during the answer. Use the third person.
    
        Answer without mentioning the Context. Use it but don't refer to it in the text.
    
        To answer, use max 300 word.
    
        Create a Justification from the sentences given.
        Given a claim and scientific abstracts (where the topic or claim may be cited or mentioned).
        Your task is to extract verbatim sentences from the abstracts that are most relevant to the claim or show the scope/limits of what the abstracts actually report.
        (helping explain why the claim cannot be confirmed or refuted)
        
        Strict Rules:
        - Copy sentences STRICTLY VERBATIM (character-for-character / word-for-word) from the provided abstracts.
        - DO NOT paraphrase, rewrite, summarize, translate, complete, or alter punctuation or wording in any way.
        - DO NOT invent sentences that are not explicitly present in the abstracts.
        - Only include sentences that actually appear in the text. If no sentence in an abstract is relevant or useful, write the absence of relevant information.
        - Do not add explanations outside the requested sections.
        - Keep the output format strictly as specified below
        
        Output format:
        <exact verbatim sentence>
        
        (or write a sentence of minimal length explaining the absence of relevant information for the claim, if no sentence in the abstracts is relevant or useful. Use the claim to guide the explanation, but do not invent information.)
        
        [/INST] <</SYS>>
        
        [INST] Claim: {query} [/INST]
        [INST] Abstracts:
        """

    prompt_true = f"""<<SYS>> [INST]

            You are a helpful, respectful and honest Doctor. Always answer as helpfully as possible using the context text provided.

            Use the information in Context.

            Elaborate the Context to generate a new information.

            Use only the knowledge in Context to answer.

            Be strict about using only the information provided in the Context from the scientific papers.

            Answer describing in a scientific way. Be formal during the answer. Use the third person.

            Answer without mentioning the Context. Use it but don't refer to it in the text.

            To answer, use max 300 word.

            Create a Justification from the sentences given.
            Given a claim and scientific abstracts.
            Your task is to use the information in the evidence to generate a justification that exclusively supports the claim.
            (helping explain why the claim can be confirmed)

            Strict Rules:
                - DO NOT invent data that are not explicitly present in the evidence.
                - Don not exactly copy the evidence
                - Strictly stay within the information provided by the evidence.
                - Do not add explanations outside the requested sections.
                - Keep the output format strictly as specified below

            Output format:
            <exact verbatim sentence>

            [/INST] <</SYS>>

            [INST] Claim: {query} [/INST]
            [INST] Evidence from scientific abstracts:
            """

    prompt_false = f"""<<SYS>> [INST]

                You are a helpful, respectful and honest Doctor. Always answer as helpfully as possible using the context text provided.

                Use the information in Context.

                Elaborate the Context to generate a new information.

                Use only the knowledge in Context to answer.

                Be strict about using only the information provided in the Context from the scientific papers.

                Answer describing in a scientific way. Be formal during the answer. Use the third person.

                Answer without mentioning the Context. Use it but don't refer to it in the text.

                To answer, use max 300 word.

                Create a Justification from the sentences given.
                Given a claim and scientific abstracts.
                Your task is to use the information in the evidence to generate a justification that exclusively refutes the claim.
                (helping explain why the claim can be rejected)

                Strict Rules:
                    - DO NOT invent data that are not explicitly present in the evidence.
                    - Don not exactly copy the evidence
                    - Strictly stay within the information provided by the evidence.
                    - Do not add explanations outside the requested sections.
                    - Keep the output format strictly as specified below

                Output format:
                <exact verbatim sentence>

                [/INST] <</SYS>>

                [INST] Claim: {query} [/INST]
                [INST] Evidence from scientific abstracts:
                """

    if label == "SUPPORT":
        llm_prompt_template = prompt_true
    elif label == "CONTRADICT":
        llm_prompt_template = prompt_false
    else:
        llm_prompt_template = prompt_nei

    return llm_prompt_template



# -------------------------------------------- MAIN FUNCTION ------------------------------------------- #
if __name__ == "__main__":

    claims_path = DATA_PATH / "claims_train.jsonl"

    # ------------------------------------------- CLAIM INPUT ------------------------------------------- #
    records = get_claim_records(claims_path, CLAIMS_NUMBER)

    results={}
    justifications = {}

    for record_index, record in tqdm(enumerate(records), total=len(records), desc="Processing claims"):

        # ------------------------------------------- ESTRAZIONE ABSTRACTS ------------------------------------------- #


        abstracts = get_abstracts_for_claims(claims_path, CLAIMS_NUMBER)

        # ------------------------------------------- PROCESSING OF THE TOP ABSTRACTS -------------------------------------------#
        if NVIDIA or LOCAL:
            prompt_template = llm_reasoning_template(record["claim"], record["true_label"])

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
            for i, abstract in enumerate(abstracts[record["id"]], start=1):
                prompt_template += f"{abstract} ; "
            prompt_template += "[/INST]"

            answer = gemini_generate(prompt_template, model=LLM_MODEL)

        # ------------------------------------------- JUSTIFICATION GENERATION ------------------------------------------- #

        justifications[record["id"]] = answer

    # ------------------------------------------- SAVE JUSTIFICATIONS ------------------------------------------- #
    save_justifications(
        input_path=DATA_PATH / "claims_train.jsonl",
        output_path=DATA_PATH / "claims_train_all_justified.jsonl",
        justifications=justifications,
    )