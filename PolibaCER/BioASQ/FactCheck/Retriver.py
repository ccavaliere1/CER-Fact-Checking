from pathlib import Path
import dill as pickle
from retriv import SparseRetriever


DATA_PATH = Path(__file__).resolve().parents[1] / "data"
corpus_path = DATA_PATH / "PubMed_p205.csv"


sr = SparseRetriever(
  index_name="pubmed-index",
  model="bm25",
  min_df=10,
  tokenizer="whitespace",
  stemmer="english",
  stopwords="english",
  do_lowercasing=True,
  do_ampersand_normalization=True,
  do_special_chars_normalization=True,
  do_acronyms_normalization=True,
  do_punctuation_removal=True,
)

import time
start = time.time()

sr = sr.index_file(
  path=corpus_path,
  show_progress=True,
  callback=lambda doc: {
    "id": doc["PMID"],
    "text": doc["AbstractText"],
    }
  )

with open(DATA_PATH/ 'PubMed_SparseRetriever', 'wb') as file:
    pickle.dump(sr, file)