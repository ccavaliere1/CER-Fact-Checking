import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data"

with open(DATA_PATH / "BioASQ-train-yesno-7b.json", "r") as f:
    bioasq_dataset = json.load(f)

bioasq_questions = list()
bioasq_answers = list()

for item in bioasq_dataset["data"][0]["paragraphs"]:
    question = item["qas"][0]["question"]
    answer = item["qas"][0]["answers"]
    if question not in bioasq_questions:
        bioasq_questions.append(question)
        bioasq_answers.append(answer)
