def llm_reasoning_template(query):
    """
    Generate a template for the prompt used for justification generation by the LLM model.

    Args:
        query (str): Claim to fact-check.

    Returns:
        str: Reasoning template for the claim.
    """
    llm_reasoning_prompt = f"""<<SYS>> [INST]

    You are a helpful, respectful and honest Doctor. Always answer as helpfully as possible using the context text provided.

    Use the information in Context.

    Elaborate the Context to generate a new information.

    Use only the knowledge in Context to answer.

    Answer describing in a scentific way. Be formal during the answer. Use the third person.

    Answer without mentioning the Context. Use it but don't refer to it in the text.

    To answer, use max 300 word.

    Create a Justification from the sentences given.

    Use the structure: Justification: The claim is (label) because... (don't use the word "context")

    Write as an online doctor to create the Justification.

    After, give some sentences from Context from scientific papers: that supports the label and reject the label.

    Supporting sentences from abstracts:
    information sentence from abstract_1: 
    information sentence from abstract_2: 
    ..
    Refusing sentences from abstracts:
    information sentence from abstract_1: 
    information sentence from abstract_2: 
    ..
    Add where it comes from (abstract_1, abstract_2, abstract_3, abstract_4, abstract_5)

    With the answer, gives a line like: "Label:". Always put Label as first. After Label, give the Justification.
    The justification will be always given as Justification: 
    Label can be yes, no, NEI, where yes: claim is true. no: claim is false. NEI: not enough information.
    The Label will be chosen with a voting system of support/refuse before.
    Important: The supporting and refusing sentences MUST be copied verbatim (word-for-word) from the provided abstracts. Do not paraphrase, summarize, or alter any words.

    Please write the Justification and Note in the same language as the user's question, but strictly keep the structural section titles ('Label:', 'Justification:', 'Supporting sentences from abstracts:', 'Refusing sentences from abstracts:', 'Note:') in English.

    [/INST] <</SYS>>

    [INST] Question: {query} [/INST]
    [INST] Context from scientific papers: 
    """

    return llm_reasoning_prompt


def reason_model_response(main_global, claim, top_abstracts, client, llm_model):
    """
    Generate a response from the reasoning model based on the claim and abstracts.

    Args:
        main_global: globals() dictionary containing global variables.
        claim (str): Claim to fact-check.
        top_abstracts (list): List of top abstracts.
        client (OpenAI): OpenAI client.
        llm_model (str): Name of the LLM model.
    """

    prompt_template = llm_reasoning_template(claim)

    # Add the abstracts to the prompt
    for i in range(1, len(top_abstracts)):
        prompt_template += f"{main_global[f'abstract_{i}']} ; "
        prompt_template += f"{main_global[f'abstract_{len(top_abstracts)}']} [/INST]"

    completion = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt_template}],
        temperature=0.1,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )

    # Collect the response
    answer = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            answer += chunk.choices[0].delta.content

    return answer