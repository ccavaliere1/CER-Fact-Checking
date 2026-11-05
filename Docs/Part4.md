Per ottenere le metriche del paper devi trasformare il sistema in una pipeline di valutazione automatica. Non devi calcolare precision, recall e F1 sulle spiegazioni generate dall’LLM, ma sulle **predizioni finali confrontate con le label corrette dei dataset**.

## 1. Costruisci il dataset di valutazione

Per ogni esempio devi avere almeno:

```text
claim
gold_label
predicted_label
```

Per esempio:

| Claim | Gold label | Predicted label |
|---|---|---|
| “Masks reduce infections...” | SUPPORT | SUPPORT |
| “Homeopathy cures dizziness...” | CONTRADICT | NEI |
| “...” | NEI | SUPPORT |

Le label devono essere normalizzate prima del calcolo. Ad esempio, per SciFact:

```text
SUPPORT
CONTRADICT
NEI
```

Per HealthFC devi verificare il significato esatto delle label numeriche `0`, `1`, eventualmente `2`, perché non devi assumere automaticamente che `0` significhi “vero”.

Inoltre devi mantenere separati:

- train;
- validation/dev;
- test.

Le metriche finali vanno calcolate sul **test set**, senza usare le label del test durante il retrieval supervisionato, il fine-tuning o la scelta dei parametri.

## 2. Genera le predizioni

Per ogni claim del test set:

1. recupera gli abstract/documenti;
2. passa al classificatore soltanto le evidenze recuperate;
3. ottieni una predizione tra le classi previste;
4. salva la predizione su file.

Un file di output dovrebbe avere una struttura simile:

```csv
id,gold_label,predicted_label
1,SUPPORT,SUPPORT
2,CONTRADICT,NEI
3,NEI,NEI
```

Se il sistema usa un LLM, devi salvare anche:

```text
claim
retrieved_document_ids
prompt
LLM response
predicted_label
```

In questo modo puoi ricalcolare le metriche senza richiamare nuovamente il modello remoto.

## 3. Calcola precision, recall e F1

Per una singola classe:

\[
Precision = \frac{TP}{TP + FP}
\]

La precision risponde alla domanda:

> Tra gli esempi classificati dal sistema come questa classe, quanti sono corretti?

\[
Recall = \frac{TP}{TP + FN}
\]

La recall risponde alla domanda:

> Tra tutti gli esempi che appartengono realmente a questa classe, quanti sono stati trovati?

\[
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
\]

L’F1 combina precision e recall.

Con tre classi, come in SciFact, devi calcolare le metriche per ogni classe:

| Classe | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUPPORT | ... | ... | ... |
| CONTRADICT | ... | ... | ... |
| NEI | ... | ... | ... |

Poi devi usare la stessa aggregazione adottata dal paper.

### Macro-F1

Calcola l’F1 di ogni classe e poi la media semplice:

\[
F1_{macro} =
\frac{F1_{SUPPORT} + F1_{CONTRADICT} + F1_{NEI}}{3}
\]

Tratta tutte le classi allo stesso modo, anche se sono sbilanciate.

### Micro-F1

Somma tutti i true positive, false positive e false negative prima di calcolare la metrica. È più influenzato dalle classi numericamente più frequenti.

### Weighted-F1

Calcola la media degli F1 pesata per il numero di esempi di ogni classe.

Non puoi confrontare correttamente il tuo `macro-F1` con un `weighted-F1` riportato nel paper. Devi verificare nelle tabelle o nella sezione di valutazione quale variante viene usata.

Un calcolo possibile è:

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=["SUPPORT", "CONTRADICT", "NEI"],
    average=None,
    zero_division=0,
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["SUPPORT", "CONTRADICT", "NEI"],
        average="macro",
        zero_division=0,
    )
)

print(classification_report(
    y_true,
    y_pred,
    labels=["SUPPORT", "CONTRADICT", "NEI"],
    digits=4,
    zero_division=0,
))
```

Il risultato da riportare dovrebbe includere almeno:

```text
precision_macro
recall_macro
f1_macro
precision/recall/f1 per classe
confusion matrix
numero di esempi
```

## 4. Devi valutare separatamente retrieval e classificazione

Nel paper CER non è soltanto un classificatore. Prima deve trovare l’evidenza scientifica. Pertanto la valutazione deve essere divisa in due parti.

### A. Valutazione del retrieval

Per ogni claim devi salvare i documenti recuperati:

```text
gold_doc_ids
retrieved_doc_ids
```

Poi puoi calcolare, ad esempio:

- Recall@1;
- Recall@5;
- Recall@10;
- Precision@k;
- eventualmente MRR o nDCG, se presenti nel paper.

Per esempio:

```text
Recall@5 =
numero di claim per cui almeno un documento gold è nei primi 5
/
numero totale di claim
```

Per SciFact puoi anche valutare il recupero delle singole frasi di evidenza, non soltanto del documento.

### B. Valutazione della classificazione

Dopo il retrieval, calcoli precision, recall e F1 sulle label finali:

```text
SUPPORT
CONTRADICT
NEI
```

Devi distinguere almeno due condizioni:

| Condizione | Evidenza usata |
|---|---|
| Oracle evidence | evidenza annotata nel dataset |
| Retrieved evidence | evidenza trovata dal sistema |

La seconda è quella più realistica e misura il sistema completo. La prima serve per capire se gli errori derivano dal classificatore oppure dal retrieval.

Se usi le evidenze gold per classificare il test set, non stai valutando CER end-to-end: stai valutando soltanto il classificatore in condizioni ideali.

## 5. Con cosa devi confrontare i risultati

Il confronto corretto è con i risultati riportati nel paper per:

1. lo stesso dataset;
2. lo stesso split;
3. la stessa attività;
4. la stessa metrica;
5. lo stesso numero di classi;
6. lo stesso tipo di evidenza.

Il confronto dovrebbe avere una tabella di questo tipo:

| Metodo | Dataset | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Baseline 1 riportata nel paper | SciFact | ... | ... | ... |
| Baseline 2 riportata nel paper | SciFact | ... | ... | ... |
| CER | SciFact | ... | ... | ... |
| Tua replica di CER | SciFact | ... | ... | ... |

I baseline da usare sono quelli riportati nelle tabelle sperimentali del paper e nei lavori precedenti citati dagli autori per ciascun dataset. Non devi confrontare il tuo risultato con:

- una semplice maggioranza di classe, se il paper non la usa;
- l’output della demo su pochi claim;
- risultati ottenuti su uno split diverso;
- metriche calcolate con un’altra definizione di F1;
- una versione diversa del dataset;
- un retrieval PubMed diverso da quello usato nella tabella.

Se il paper presenta più configurazioni, devi riportarle separatamente, ad esempio:

```text
CER - sparse retrieval
CER - dense retrieval
CER - sparse + dense retrieval
CER - classificatore senza LLM
CER - sistema completo
```

Il confronto più importante è quello tra il **sistema completo CER** e i baseline sullo stesso benchmark.

## 6. Non confrontare direttamente il risultato della demo con il paper

Nel codice attuale la demo usa:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

per il retrieval, recupera `top_k=5` abstract, usa un LLM remoto per la spiegazione e poi:

```python
MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33
```

per la classificazione zero-shot.

Questo output non è automaticamente il risultato sperimentale riportato nel paper. Per poterlo confrontare devi sapere se le tabelle del paper usano gli stessi:

- modelli;
- prompt;
- dataset;
- split;
- parametri;
- documenti recuperati;
- criteri di classificazione.

In particolare, una spiegazione corretta dell’LLM non implica necessariamente una label corretta, e una label corretta non dimostra che l’evidenza sia stata recuperata correttamente.

## 7. Come presentare il risultato finale

Per ogni dataset dovresti produrre almeno:

```text
Dataset: SciFact
Split: test
N esempi: ...
Retrieval: dense
Encoder: ...
Top-k: 5
Classifier: ...
Metric: macro-F1

SUPPORT:
  precision = ...
  recall    = ...
  F1        = ...

CONTRADICT:
  precision = ...
  recall    = ...
  F1        = ...

NEI:
  precision = ...
  recall    = ...
  F1        = ...

Macro precision = ...
Macro recall    = ...
Macro F1        = ...
```

Poi aggiungi una tabella comparativa con i metodi del paper.

La conclusione deve distinguere chiaramente:

- **replica riuscita**, se il risultato è vicino a quello del paper con la stessa configurazione;
- **replica parziale**, se riesci a riprodurre il processo ma non tutti i componenti;
- **risultato non confrontabile**, se cambiano dataset, split, modelli o definizione delle metriche.

Nel tuo caso, il passaggio fondamentale è creare uno script di valutazione separato da `app.py`. L’applicazione attuale è progettata per una dimostrazione interattiva e non salva automaticamente `y_true`, `y_pred`, documenti recuperati e metriche. Senza questi elementi non puoi confrontare rigorosamente i risultati con le tabelle del paper.