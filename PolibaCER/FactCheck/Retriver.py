import faiss

# ----------------------------------------------- PUBMED RETRIEVER ------------------------------------------------ #
def retrieve_pubmed_top_abstracts(claim, model, index, pmids, data, top_k=5):
    """
    Retrieve the top abstracts from the FAISS index for a given claim.

    Args:
        claim (str): Claim to fact-check.
        model (SentenceTransformer): Sentence transformer model for encoding text.
        index (faiss.IndexFlatIP): FAISS index for similarity search.
        pmids (np.ndarray): Array of PMIDs for the abstracts.
        data (pd.DataFrame): DataFrame containing the abstract data.
        top_k (int): Number of top abstracts to retrieve.

    Returns:
        list: List of tuples containing the abstract text, PMID, and distance.
    """
    # Encode the claim using the SentenceTransformer model
    claim_embedding = model.encode([claim])
    faiss.normalize_L2(claim_embedding)  # Normalize the claim embedding (with L2 norm)
    distances, indices = index.search(claim_embedding, top_k)

    # Retrieve the top abstracts based on the indices
    results = []
    for j, i in enumerate(indices[0]):
        pmid = pmids[i]
        abstract_text = data[data['PMID'] == pmid]['AbstractText'].values[0]
        distance = distances[0][j]
        results.append((abstract_text, pmid, distance))

    return results
# ----------------------------------------------- END PUBMED RETRIEVER ------------------------------------------------ #

# ----------------------------------------------- SCIFACT RETRIEVER ------------------------------------------------ #
def retrieve_scifact_top_abstracts(claim, model, index, doc_ids, documents,top_k=5,):
    """
    Recupera i top-k documenti SciFact per un claim.

    Restituisce una lista ordinata per similarità decrescente.
    """
    if not claim or not claim.strip():
        raise ValueError("Il claim non può essere vuoto.")

    if top_k < 1:
        raise ValueError("top_k deve essere almeno 1.")

    # APP VERSION
    query_embedding = model.encode([claim])
    faiss.normalize_L2(query_embedding)  # Normalize the claim embedding (with L2 norm)
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, position in zip(distances[0], indices[0]):
        doc_id = int(doc_ids[position])
        document = documents[doc_id]

        results.append(
            {
                "doc_id": doc_id,
                "title": document["title"],
                "abstract": document["text"],
                "sentences": document["sentences"],
                "distance": float(distance),
            }
        )

    return results
# ----------------------------------------------- END SCIFACT RETRIEVER ------------------------------------------------ #