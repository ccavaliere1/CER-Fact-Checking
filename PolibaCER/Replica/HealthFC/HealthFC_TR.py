from pathlib import Path
import pandas as pd
import numpy as np  # Importare NumPy
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DATA_PATH = Path(__file__).resolve().parents[0] / "data"
trainDF_path = DATA_PATH / 'TrainDS' / 'train1.csv'
valDF_path = DATA_PATH / 'TrainDS' / 'val1.csv'
testDF_path = DATA_PATH / 'TrainDS' / 'test1.csv'

#train1 sono 3 classi
train_df = pd.read_csv(trainDF_path)
val_df = pd.read_csv(valDF_path)
test_df = pd.read_csv(testDF_path)

# Convertire i DataFrame in Dataset di Hugging Face
def convert_to_dataset(df):
    return Dataset.from_pandas(df)

# Supponiamo che train_df, val_df, e test_df siano già definiti
train_dataset = convert_to_dataset(train_df)
val_dataset = convert_to_dataset(val_df)
test_dataset = convert_to_dataset(test_df)

# Creare un DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# Preprocessare e tokenizzare
model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    # Concatenare Claim e Justification e mappare le etichette
    texts = [claim + " " + justification for claim, justification in zip(examples['Claim'], examples['Justification'])]
    labels = [1 if label == 'yes' else 0 for label in examples['True Label']]
    encodings = tokenizer(texts, padding="max_length", truncation=True)
    encodings['labels'] = labels
    return encodings

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Configurare il modello
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Definire gli argomenti di training
training_args = TrainingArguments(
    output_dir=f"{DATA_PATH / 'Results'}",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.20,
    gradient_accumulation_steps=8,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_strategy="epoch"
)

# Definire la funzione di metriche
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision = precision_score(p.label_ids, preds, average='macro')
    recall = recall_score(p.label_ids, preds, average='macro')
    f1 = f1_score(p.label_ids, preds, average='macro')
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Creare il Trainer con specifica della funzione di perdita
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    compute_metrics=compute_metrics
)

# Addestrare il modello
trainer.train()

# Valutazione del modello sul set di test
metrics = trainer.evaluate(encoded_dataset['test'])
print("Test set evaluation metrics:", metrics)