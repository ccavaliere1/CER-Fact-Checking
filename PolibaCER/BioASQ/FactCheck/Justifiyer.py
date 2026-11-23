import re
from pathlib import Path
import pandas as pd
from PolibaCER.BioASQ.FactCheck.Reasoning import get_completion

DATA_PATH = Path(__file__).resolve().parents[1] / "data"
bioASQ_path = DATA_PATH / "BioASQ_sentences_true_label.csv"
health_fc = DATA_PATH / "Health_fc_sentences_true_label.csv"

dataframe = pd.read_csv(bioASQ_path)

# -------------------------------------------- LLM RETRIVER_MODEL SETUP ------------------------------------------- #
def add_to_dataframe(df, query, Risposta):
    # Trova la label nella risposta usando una regex
    label_match = re.search(r"Label: (\w+)", Risposta)
    if label_match:
        label = label_match.group(1)  # Estrai la label

        justification = re.sub(r'^Justification:\s*', '', Risposta[label_match.end():].strip())

        claim = query.strip()  # Il claim è semplicemente la query data
        new_row = pd.DataFrame([[claim, label, justification]], columns=["Claim", "Predicted label", "Justification"])
        df = pd.concat([df, new_row], ignore_index=True)
        return df

# -------------------------------------------- JUSTIFICATION ------------------------------------------- #

def justify_dataframe(dataframe, start_index=1, num_rows=2, llm_model="bartowski/Qwen2.5-7B-Instruct-GGUF:Q6_K"):

    for index in range(start_index, start_index + num_rows):
        query = dataframe.loc[index, 'Frase']
        results = dataframe.loc[index, 'Sentences Concatenate']

        # Costruisci il prompt template con query e results prelevati
        prompt_template = f'''[INST]  <<SYS>>

        You are a helpful, respectful and honest Doctor. Always answer as helpfully as possible using the context text provided. Try to give an explanation based on {results} to the question {query}. elaborate the context to generate a new information.

        Use only the knowledge in results to answer.

        Answer describing in a scientific way. Be formal during the answer. Use the third person.

        Answer without mentioning the context. Use it but don't refer to it in the text

        to answer, use max 200 word

        with the answer, gives a line like: "Label:". Always put Label as first. After Label, give the justification
        Label can be yes, no, NEI, where yes: claim is true. no: claim is false. NEI: not enough information.
        <<SYS>>

        """CONTEXT:/n/n {results}/n

        Question: {query} [/INST]

        '''

        responses = get_completion(prompt_template, model=llm_model, max_tokens=400)
        Risposta = ''.join([str(r) for r in responses])

        # Stampa la risposta per ogni iterazione
        print("")
        print(Risposta)
        try:
            dfj
        except NameError:
            dfj = pd.DataFrame(columns=["Claim", "Predicted label", "Justification"])
        # Esempio di utilizzo
        dfj = add_to_dataframe(dfj, query, Risposta)

    return dfj

if __name__ == "__main__":
    # Esegui la funzione justify_dataframe con i parametri desiderati
    dfj = justify_dataframe(dataframe, start_index=1, num_rows=2, llm_model="bartowski/Qwen2.5-7B-Instruct-GGUF:Q6_K")
    print(dfj)