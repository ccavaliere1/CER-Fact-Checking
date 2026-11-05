**PubMed, SciFact e HealthFC non sono tre alternative equivalenti**: hanno ruoli diversi nella pipeline CER.

```text
PubMed   = fonte/corpus di articoli scientifici
SciFact  = benchmark annotato per valutare fact-checking scientifico
HealthFC = benchmark annotato per valutare fact-checking sanitario
```

## PubMed

**PubMed è una risorsa bibliografica**, non principalmente un dataset di classificazione.

Contiene record di pubblicazioni biomediche, tra cui:

- PMID;
- titolo;
- abstract;
- autori;
- rivista;
- data;
- metadati bibliografici.

Nel sistema CER PubMed svolge il ruolo di **base documentale da cui recuperare l’evidenza**:

```text
claim
   ↓
ricerca su PubMed
   ↓
abstract/articoli rilevanti
   ↓
LLM e classificatore
```

Nel repository, questo ruolo è rappresentato soprattutto da:

```text
data\parte_205.csv
data\abstract_embeddings.npy
data\pmids.npy
data\faiss_index.index
```

Il CSV contiene PMID e abstract; gli altri file sono strutture derivate per il retrieval semantico.

PubMed non ti dice direttamente se un claim è:

```text
vero
falso
NEI
```

Questa decisione deve essere ottenuta confrontando il claim con le evidenze recuperate.

## SciFact

SciFact è un **benchmark supervisionato** progettato per valutare la verifica di claim scientifici.

Contiene:

- claim;
- label;
- documenti scientifici;
- frasi di evidenza;
- split train/dev/test.

Le label sono tipicamente:

```text
SUPPORT
CONTRADICT
NEI
```

Il corpus SciFact contiene già i documenti da usare per il benchmark. Per la valutazione standard devi quindi usare:

```text
data\SciFact - data\corpus.jsonl
```

e non il corpus PubMed locale.

La pipeline SciFact è:

```text
claim SciFact
   ↓
retrieval su corpus.jsonl
   ↓
selezione delle frasi
   ↓
SUPPORT / CONTRADICT / NEI
   ↓
precision, recall, F1
```

SciFact serve quindi a valutare in modo controllato:

1. se il sistema recupera il documento corretto;
2. se individua le frasi corrette;
3. se classifica correttamente il claim.

## HealthFC

HealthFC è anch’esso un **benchmark supervisionato**, ma è orientato alla comunicazione sanitaria e al fact-checking di contenuti divulgativi.

Nel file del progetto trovi claim, spiegazioni, frasi di supporto e label:

```text
data\HealtFc - data.csv
```

Esempi di contenuti sono domande su:

- mascherine;
- farmaci;
- alimentazione;
- rimedi alternativi;
- prevenzione;
- trattamenti sanitari.

HealthFC è quindi più vicino al problema:

> una persona legge o pubblica un’informazione sanitaria: quanto è affidabile?

Rispetto a SciFact, contiene più spesso claim legati a contenuti di divulgazione e articoli di fact-checking, invece di claim scientifici costruiti a partire dalla letteratura accademica.

## Differenze principali

| Aspetto | PubMed | SciFact | HealthFC |
|---|---|---|---|
| Tipo | Database/corpus bibliografico | Benchmark annotato | Benchmark annotato |
| Funzione | Fornire articoli e abstract | Valutare claim scientifici | Valutare claim sanitari |
| Contiene label finali? | No, in generale | Sì | Sì |
| Contiene evidenze gold? | No, non nel formato benchmark | Sì | Frasi/spiegazioni annotate |
| Uso nel sistema | Retrieval | Training/evaluation | Training/evaluation |
| Identificativi | PMID | `doc_id` SciFact | ID/riga del dataset e URL |
| Corpus da usare | Record PubMed | `corpus.jsonl` | Dataset HealthFC più fonti associate |
| Metriche | qualità del retrieval, eventualmente | retrieval + classification | classification, eventualmente explanation |

## Come si collegano nella pipeline CER

Il ruolo ideale è:

```text
PubMed
  fornisce le evidenze scientifiche al sistema operativo

SciFact
  verifica sperimentalmente retrieval e classificazione su un benchmark controllato

HealthFC
  verifica il comportamento su claim sanitari divulgativi
```

In termini di sviluppo:

### Esperimento SciFact

```text
claims_test.jsonl
      +
corpus.jsonl
      ↓
retrieval
      ↓
classificazione
      ↓
metriche SciFact
```

### Esperimento HealthFC

```text
HealthFC
      ↓
claim + label gold
      ↓
retrieval delle fonti richieste dal protocollo
      ↓
classificazione
      ↓
metriche HealthFC
```

### Demo CER con PubMed

```text
claim inserito dall’utente
      ↓
FAISS su abstract PubMed
      ↓
LLM
      ↓
classificatore
      ↓
True / False / NEI + fonti PubMed
```

## Perché non puoi sostituirli direttamente

Se utilizzi PubMed al posto di SciFact, cambi:

- il corpus;
- gli identificativi dei documenti;
- la copertura documentale;
- il preprocessing;
- le evidenze disponibili;
- la distribuzione dei risultati.

Di conseguenza, non puoi confrontare direttamente quel risultato con la metrica ufficiale SciFact.

Analogamente, HealthFC non è semplicemente un sottoinsieme di PubMed: le sue label e spiegazioni sono legate a un’attività di fact-checking sanitario, non alla sola presenza di un abstract scientifico.

## Un esempio

Supponiamo il claim:

> “Paxlovid riduce il rischio di ospedalizzazione nei pazienti ad alto rischio.”

### Con PubMed

CER cerca articoli pertinenti e usa i loro abstract per produrre una risposta.

Output possibile:

```text
Label: True
Evidence: abstract PubMed associati ai PMID recuperati
```

PubMed fornisce i documenti, ma non fornisce necessariamente una label `True`.

### Con SciFact

Il claim è già associato a:

- una label gold;
- uno o più documenti gold;
- eventuali frasi gold.

Il tuo sistema deve recuperare il documento dal corpus SciFact e confrontare la propria predizione con l’annotazione.

### Con HealthFC

Il claim può essere associato a:

- una valutazione editoriale;
- una spiegazione;
- frasi di supporto;
- un verdetto sanitario.

La verifica confronta la predizione del sistema con quel verdetto.

## Quindi, nella tua replica

Devi distinguere chiaramente due obiettivi.

### Riprodurre le metriche dei benchmark

Usa:

- `corpus.jsonl` per SciFact;
- il file HealthFC e il protocollo ufficiale HealthFC;
- gli split e le label originali.

Qui PubMed non è obbligatorio e normalmente non va usato come corpus sostitutivo.

### Riprodurre il comportamento applicativo di CER

Usa:

- corpus PubMed;
- `parte_205.csv`;
- indice FAISS;
- modello di embedding;
- LLM e classificatore della demo.

Qui puoi verificare se CER recupera evidenze e produce una decisione per claim reali, ma non stai necessariamente replicando le metriche ufficiali SciFact o HealthFC.

**In sintesi:** PubMed è la sorgente di conoscenza usata dal sistema; SciFact e HealthFC sono dataset annotati usati per misurare quanto il sistema funzioni. SciFact valuta soprattutto il fact-checking di claim scientifici con evidenze documentali e frasali esplicite, mentre HealthFC valuta claim sanitari divulgativi con verdict e spiegazioni editoriali.