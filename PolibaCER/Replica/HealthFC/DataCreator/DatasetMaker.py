from pathlib import Path
import pandas as pd


def get_healthfc_data():
    DATA_PATH = Path(__file__).resolve().parents[1] / "data"

    hfc_df = pd.read_csv(f"{DATA_PATH}/healthFC_annotated.csv")
    hfc_questions = hfc_df.en_claim.tolist()
    hfc_labels = hfc_df.label.tolist()

    hfc_data = {"questions": hfc_questions, "answers": hfc_labels}
    return hfc_data