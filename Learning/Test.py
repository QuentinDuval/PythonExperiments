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

    bi_gram = NGramTokenizer(tokenizer, count=2)
    print(bi_gram.tokenize("fix a nasty memory corruption"))

    corpus = CommitMessageCorpus.from_split('train')
    vocabulary = Vocabulary.from_corpus(corpus=corpus, tokenizer=bi_gram, min_freq=5, add_unknowns=True)
    print(len(vocabulary))
    print(vocabulary)

    '''
    corpus = CommitMessageCorpus.from_file('resources/perforce_cl_unsupervised.txt', keep_unclassified=True)

    tokenizer = NltkTokenizer()
    for fix_description in corpus.get_unclassified()[30:60]:
        print(fix_description.strip())
        print(">>>", " ".join(tokenizer(fix_description)))
        print()

    vocab = Vocabulary.from_corpus(corpus, tokenizer=tokenizer, min_freq=2, add_unknowns=True)
    print(vocab.words())
    print(len(vocab))
    '''


# test_tokenization()
# test_model_1()
# test_model_2(split_seed=0)
# test_model_3(split_seed=0)
# test_model_4(split_seed=0)
# test_model_5(split_seed=0)
# test_model_6(split_seed=0)

"""
move CollateralAgreement to lib/folder/folder
> refactor (83.93%)

quantity was wrong
> fix (64.35%)

add tab in collateral screen
> feat (60.22%)

purpose of life
> fix (40.64%)

move things around
> refactor (48.95%)

rework collateral screen
> feat (39.18%)

introduce crash on purpose
> fix (50.43%)

memory regression
> fix (53.33%)

memory corruption
> fix (46.9%)

cleaning
> refactor (54.68%)

add new tab in collateral screen
> feat (73.13%)

optimize performance of getQuantity
> refactor (46.36%)

optimize performance
> feat (42.98%)

optimize performance of collateral screen
> feat (55.41%)
"""

# run_unsupervised_learning(embedding_size=20)
# test_unsupervised_learning()
