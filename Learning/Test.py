from Learning.Classifier1 import *
from Learning.Classifier2 import *
from Learning.Classifier3 import *
from Learning.Classifier4 import *
from Learning.Classifier5 import *
from Learning.Classifier6 import *
from Learning.Tokenizer import *
from Learning.Vocabulary import *


def test_tokenization():
    tokenizer = NltkTokenizer()
    for w in ["doubleAverageA1DivA2KAdjust", "COMDI-67", "FPB-75", "DEF0889790", "getQuantityOfTrade", "getQuantity", "get_quantity"]:
        print(tokenizer.clean_token(w))

    corpus = CommitMessageCorpus.from_file('resources/perforce_cl_unsupervised.txt', keep_unclassified=True)

    tokenizer = NltkTokenizer()
    for fix_description in corpus.get_unclassified()[30:60]:
        print(fix_description.strip())
        print(">>>", " ".join(tokenizer(fix_description)))
        print()

    vocab = Vocabulary.from_corpus(corpus, tokenizer=tokenizer, min_freq=2, add_unknowns=True)
    print(vocab.words())
    print(len(vocab))


# test_tokenization()
# test_model_1()
# test_model_2(split_seed=0)
# test_model_3(split_seed=0)
# test_model_4(split_seed=0)
# test_model_5(split_seed=0)
# test_model_6(split_seed=0)

"""
> introduce crash on purpose
fix
> move CollateralAgreement class in traderep module
refactor
> extract method X from Y
refactor
> quantity was wrong
fix
> add new menu in Collateral agreement screen
feat
"""

# test_model_3_interactive()
