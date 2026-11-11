def input_claim_text(test_claim=None):
    if test_claim is not None:
        claim = test_claim
    else:
        claim = input("Inserisci un claim: ")

    return claim