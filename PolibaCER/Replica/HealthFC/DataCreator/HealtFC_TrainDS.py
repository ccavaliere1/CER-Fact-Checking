from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data"

df = pd.read_csv(DATA_PATH / "Results" / "HealthFC_ZS_final_results_no_classfied.csv")

df = pd.DataFrame({
    "Claim": df["Claim"],
    "Justification": [
        f'"prediction: {predicted_label}" for: justification used for the prediction "{justification}" with sentencies {sentences}'
        for predicted_label, justification, sentences in zip(
            df["Predicted Label"],
            df["Justification"],
            df["Sentence_concatenate"],
        )
    ],
    "True Label": df["True Label"],
})


# Prima, suddividiamo in train+validation e test
train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['True Label'], random_state=42)

# Poi, suddividiamo train+validation in train e validation
train_df, val_df = train_test_split(train_val_df, test_size=0.25, stratify=train_val_df['True Label'], random_state=42)
# Nota: 0.25 * 0.8 = 0.2, quindi il set di validazione sarà il 20% del totale

trainDF_path = DATA_PATH / 'TrainDS' / 'train1.csv'
valDF_path = DATA_PATH / 'TrainDS' / 'val1.csv'
testDF_path = DATA_PATH / 'TrainDS' / 'test1.csv'

# Salva il DataFrame in un file CSV
train_df.to_csv(trainDF_path, index=False)
val_df.to_csv(valDF_path, index=False)
test_df.to_csv(testDF_path, index=False)
print(f"DataFrame salvato con successo in: {trainDF_path}")