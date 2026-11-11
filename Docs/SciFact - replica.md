Per replicare i risultati di **SciFact** devi realizzare una pipeline sperimentale separata dalla demo Streamlit. L’obiettivo è riprodurre questa sequenza:

```text
claim SciFact
   ↓
retrieval dei documenti dal corpus SciFact
   ↓
selezione delle frasi di evidenza
   ↓
classificazione SUPPORT / CONTRADICT / NEI
   ↓
calcolo delle metriche
```

## 1. Di cosa hai bisogno

### Dataset ufficiale SciFact

Nel repository hai già i file principali:

```text
data\SciFact - data\claims_train.jsonl
data\SciFact - data\claims_dev.jsonl
data\SciFact - data\claims_test.jsonl
data\SciFact - data\corpus.jsonl
```

Ti servono anche i file eventualmente presenti nella directory:

```text
data\SciFact - data\cross_validation
```

ma devi verificare nel paper se gli autori usano lo split standard train/dev/test oppure la cross-validation.

Il corpus contiene documenti nel formato:

```json
{
  "doc_id": 4983,
  "title": "...",
  "abstract": ["frase 1", "frase 2", "..."]
}
```

I claim contengono invece le annotazioni:

```json
{
  "id": 3,
  "claim": "...",
  "evidence": {
    "14717500": [
      {
        "sentences": [2, 5],
        "label": "SUPPORT"
      }
    ]
  },
  "cited_doc_ids": [14717500]
}
```

Da questi campi ricavi:

- testo del claim;
- documenti rilevanti;
- frasi gold;
- label `SUPPORT` o `CONTRADICT`;
- label `NEI` quando non esiste evidenza annotata.

### Ambiente software

Ti servono almeno:

- Python 3.11;
- `pandas`;
- `numpy`;
- `scikit-learn`;
- un motore BM25 oppure FAISS;
- `sentence-transformers`, se vuoi replicare il dense retrieval;
- il modello di classificazione indicato dal paper;
- gli eventuali modelli LLM indicati dal paper.

Le dipendenze del progetto sono già dichiarate in `pyproject.toml`, ma questo non significa che l’attuale `app.py` sia sufficiente per replicare SciFact.

## 2. Devi usare PubMed?

Per la replica standard di SciFact **no**.

Devi indicizzare:

```text
data\SciFact - data\corpus.jsonl
```

Non devi usare:

```text
data\parte_205.csv
data\faiss_index.index
data\pmids.npy
```

Questi file appartengono al corpus PubMed usato dalla demo CER. L’indice esistente non è compatibile con SciFact, perché il suo mapping è basato sui PMID, mentre SciFact usa i propri `doc_id`.

PubMed può essere usato soltanto in un esperimento aggiuntivo, per esempio:

```text
claim SciFact → retrieval PubMed → classificazione CER
```

Ma quel risultato non sarebbe direttamente confrontabile con la valutazione standard SciFact.

## 3. Cosa devi realizzare

Ti servono almeno questi componenti.

### A. Loader del dataset

Uno script deve:

1. leggere i file JSONL;
2. caricare il corpus in memoria;
3. unire titolo e abstract di ogni documento;
4. costruire il mapping `doc_id -> testo`;
5. leggere claim, evidenze e label.

La struttura interna dovrebbe essere simile a:

```python
corpus[doc_id] = {
    "title": title,
    "sentences": abstract_sentences,
    "text": title + " " + " ".join(abstract_sentences),
}
```

Per ogni claim devi produrre un record del tipo:

```python
{
    "id": claim_id,
    "claim": claim_text,
    "gold_doc_ids": [...],
    "gold_evidence": {
        doc_id: {
            "sentences": [...],
            "label": "SUPPORT"
        }
    },
    "gold_label": "SUPPORT"
}
```

### B. Retriever

Devi scegliere il metodo esatto previsto dall’esperimento del paper.

#### Baseline BM25

Indicizzi il testo di tutti i documenti SciFact e recuperi i primi `k` documenti per ogni claim.

#### Dense retrieval

Calcoli un embedding per ogni documento con il modello indicato dal paper, poi:

1. normalizzi gli embedding;
2. costruisci un indice FAISS;
3. calcoli l’embedding del claim;
4. recuperi i documenti più simili.

Non usare automaticamente `all-MiniLM-L6-v2`: è il modello presente nella demo, ma devi verificare che sia quello usato per la tabella SciFact del paper.

Il risultato del retriever deve essere salvato, per esempio:

```json
{
  "claim_id": 3,
  "retrieved_doc_ids": [14717500, 4983, 5836, 7912, 31715818],
  "scores": [0.91, 0.84, 0.81, 0.79, 0.76]
}
```

Devi congelare il valore di `k`, ad esempio `1`, `3`, `5` o `10`, in base al paper.

### C. Valutazione del document retrieval

Confronti:

```text
retrieved_doc_ids
```

con:

```text
gold_doc_ids
```

Calcoli almeno:

- Recall@1;
- Recall@5;
- Recall@10;
- eventualmente Precision@k;
- MRR o nDCG se riportati nel paper.

Per esempio:

```text
Recall@5 = 1
```

se almeno un documento gold del claim compare nei primi cinque documenti recuperati.

Devi calcolare queste metriche separatamente per `dev` e `test`, se il paper le riporta separatamente.

### D. Sentence retrieval

Una volta recuperati i documenti, devi verificare se il sistema individua le frasi corrette.

Per SciFact, le frasi gold sono indicate da:

```json
"sentences": [2, 5]
```

Devi chiarire l’indicizzazione: nelle annotazioni SciFact le frasi sono normalmente riferite alla numerazione interna dell’abstract; il tuo codice deve mantenere esattamente quel riferimento, senza modificare l’ordine delle frasi.

L’output dovrebbe contenere:

```json
{
  "claim_id": 3,
  "doc_id": 14717500,
  "gold_sentence_ids": [2, 5],
  "predicted_sentence_ids": [2, 5]
}
```

Poi calcoli precision, recall e F1 sulle frasi recuperate.

## 4. Classificazione del claim

Dopo il retrieval devi classificare il claim usando **soltanto i documenti/frasi recuperati**.

Le classi standard sono:

```text
SUPPORT
CONTRADICT
NEI
```

Devi salvare un file di predizioni:

```csv
claim_id,gold_label,predicted_label
3,SUPPORT,SUPPORT
5,CONTRADICT,NEI
7,NEI,NEI
```

Non devi usare direttamente `evidence` gold per generare la predizione nella valutazione end-to-end. L’evidenza gold serve per calcolare le metriche, non come input privilegiato al classificatore.

È utile eseguire due esperimenti distinti:

| Esperimento | Evidenza fornita al classificatore | Scopo |
|---|---|---|
| Oracle | evidenza gold | misura il classificatore isolato |
| End-to-end | evidenza recuperata | misura l’intero sistema |

Il risultato end-to-end è quello più importante per valutare CER.

## 5. Come integrare LLM e classificatore

Devi verificare nel paper la configurazione esatta usata per SciFact. Possibili componenti presenti nel progetto sono:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

```python
MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33
```

```text
meta/llama-3.2-11b-vision-instruct
```

Tuttavia non devi assumere che questa combinazione sia quella utilizzata per produrre i numeri del paper. Il paper e la demo non sono necessariamente equivalenti.

Se il paper usa un classificatore fine-tuned, devi avere:

- checkpoint del modello;
- tokenizer;
- mapping delle label;
- testo usato come input;
- hyperparameter;
- numero di epoche;
- seed;
- split di training;
- procedura di early stopping.

Se usa un LLM zero-shot o few-shot, devi avere:

- prompt esatto;
- modello esatto;
- temperatura;
- `top_p`;
- numero massimo di token;
- formato dell’output;
- regola di parsing delle label.

Il prompt presente in `app.py` può essere usato per replicare la demo, ma non va considerato automaticamente il prompt sperimentale del paper.

## 6. Calcolo delle metriche finali

Per la classificazione devi avere:

```python
y_true = [
    "SUPPORT",
    "CONTRADICT",
    "NEI",
]

y_pred = [
    "SUPPORT",
    "NEI",
    "NEI",
]
```

Poi calcoli:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_true,
        y_pred,
        labels=["SUPPORT", "CONTRADICT", "NEI"],
        digits=4,
        zero_division=0,
    )
)
```

Devi riportare:

- precision per classe;
- recall per classe;
- F1 per classe;
- macro-precision;
- macro-recall;
- macro-F1;
- eventualmente micro-F1 e weighted-F1;
- numero di esempi per classe;
- confusion matrix.

Non devi riportare soltanto un unico valore F1 senza indicare se è:

- macro;
- micro;
- weighted;
- binary.

## 7. Con cosa confrontare i risultati

Devi creare almeno quattro gruppi di risultati.

### 1. Baseline di retrieval

Per esempio:

```text
BM25
Dense retrieval
Sparse + dense retrieval
```

### 2. Baseline di classificazione

Utilizza i baseline elencati nella tabella del paper o nei lavori originali di SciFact. Devi mantenere:

- stesso dataset;
- stesso split;
- stessa definizione delle classi;
- stessa metrica.

Non confrontare una tua accuracy con un F1 del paper.

### 3. CER con evidenza gold

Serve a misurare il classificatore senza l’errore del retriever.

### 4. CER end-to-end

Serve a misurare il sistema completo:

```text
retrieval + evidence selection + classification
```

Una tabella finale dovrebbe avere una struttura simile:

| Sistema | Evidenza | Retrieval | Precision | Recall | Macro-F1 |
|---|---|---|---:|---:|---:|
| Baseline SciFact | gold/retrieved | BM25 | ... | ... | ... |
| CER | gold | oracle | ... | ... | ... |
| CER | retrieved | dense | ... | ... | ... |
| CER | retrieved | sparse+dense | ... | ... | ... |

Devi confrontare la tua riga CER con la riga CER del paper e con i baseline riportati nello stesso esperimento.

## 8. Quali file devi produrre

Per una replica verificabile dovresti realizzare una struttura simile:

```text
evaluation/
    load_scifact.py
    build_bm25.py
    build_dense_index.py
    retrieve.py
    select_evidence.py
    classify.py
    evaluate_retrieval.py
    evaluate_classification.py
    run_experiment.py
```

E generare questi output:

```text
results/
    retrieval_dev.json
    retrieval_test.json
    sentence_retrieval_test.json
    predictions_oracle.csv
    predictions_end_to_end.csv
    metrics_retrieval.json
    metrics_classification.json
    confusion_matrix.csv
    run_config.json
```

`run_config.json` dovrebbe contenere almeno:

```json
{
  "dataset": "SciFact",
  "split": "test",
  "retriever": "dense",
  "embedding_model": "...",
  "top_k": 5,
  "classifier": "...",
  "seed": 42,
  "temperature": 0.1
}
```

## 9. Quando puoi dire di aver replicato il paper

Puoi parlare di replica fedele soltanto se hai:

- lo stesso corpus;
- gli stessi split;
- lo stesso preprocessing;
- lo stesso retriever;
- lo stesso modello;
- lo stesso prompt o checkpoint;
- gli stessi parametri;
- la stessa definizione delle label;
- la stessa metrica;
- lo stesso protocollo di valutazione.

Se usi `corpus.jsonl`, un nuovo retriever e il modello presente nella demo, ottieni una **valutazione sperimentale CER su SciFact**, ma non necessariamente una riproduzione esatta dei risultati numerici del paper.

La sequenza concreta da realizzare è quindi:

```text
1. caricare corpus e claim SciFact
2. costruire un indice sul corpus SciFact
3. recuperare i top-k documenti
4. valutare Recall@k dei documenti
5. selezionare le frasi di evidenza
6. valutare precision/recall/F1 delle frasi
7. classificare SUPPORT/CONTRADICT/NEI
8. salvare y_true e y_pred
9. calcolare precision/recall/F1 per classe
10. confrontare i valori con le tabelle del paper
```

Il punto essenziale è non confondere la **valutazione del retrieval** con la **valutazione della classificazione**: sono due esperimenti diversi e devono avere predizioni, metriche e confronti separati.