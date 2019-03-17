import numpy as np

from Learning.Classifier2 import *
from Learning.Predictor import *
from Learning.Tokenizer import *
from Learning.Vocabulary import *


class CollapsedOneHotVectorizer(Vectorizer):
    """
    Vectorize a sentence by transforming each sentence to a vector filled with 1 for each word present, 0 otherwise
    """
    def __init__(self, vocabulary: Vocabulary, tokenizer):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer

    def vectorize(self, sentence):
        one_hot = np.zeros(len(self.vocabulary), dtype=np.float32)
        for token in self.tokenizer(sentence):
            token_index = self.vocabulary.word_lookup(token)
            one_hot[token_index] = 1
        return one_hot

    def get_vocabulary_len(self):
        return len(self.vocabulary)

    @classmethod
    def from_corpus(cls, corpus: CommitMessageCorpus, tokenizer, min_freq=2):
        vocabulary = Vocabulary.from_corpus(corpus=corpus, tokenizer=tokenizer, min_freq=min_freq, add_unknowns=True)
        return cls(vocabulary=vocabulary, tokenizer=tokenizer)


def test_model_3(with_bi_grams=False, split_seed=None):
    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')

    vectorizer = CollapsedOneHotVectorizer.from_corpus(training_corpus, NltkTokenizer(), min_freq=2)
    vocab_len = vectorizer.get_vocabulary_len()

    if with_bi_grams:
        vectorizer = CollapsedOneHotVectorizer.from_corpus(training_corpus,
                                                           tokenizer=BiGramTokenizer(NltkTokenizer()),
                                                           min_freq=2)
        vocab_len = vectorizer.get_vocabulary_len()


    """
    Training (max): 3779/4244 (89.04335532516494%)
    Validation (max): 371/472 (78.60169491525424%)
    ------------------------------
    Accuracy: 75.69955817378498 %
    """

    '''
    model = PerceptronModel(vocabulary_len=vocab_len, nb_classes=3)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus)
    predictor.evaluate(test_corpus=test_corpus)
    model.save('models/preceptron.model')
    '''

    """
    Training: 4064/4244 (95.75871819038643%)
    Validation: 355/472 (75.21186440677965%)
    Training (max): 3711/4244 (87.44109330819981%)
    Validation (max): 380/472 (80.50847457627118%)
    ------------------------------
    Accuracy: 75.25773195876289 %
    """

    '''
    model = DoublePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=3)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=1e-3, weight_decay=0)
    predictor.evaluate(test_corpus=test_corpus)
    '''

    """
    Training: 3659/4244 (86.21583411875588%)
    Validation: 386/472 (81.77966101694916%)
    Training (max): 3687/4244 (86.875589066918%)
    Validation (max): 386/472 (81.77966101694916%)
    ------------------------------
    Accuracy: 75.40500736377025 %
    
    ** With Bi-grams **
    Training (max): 3858/4221 (91.4001421464108%)
    Validation (max): 383/470 (81.48936170212767%)
    ------------------------------
    Accuracy: 76.55786350148368 %
    """

    '''
    model = DoublePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=40, nb_classes=3, drop_out=0.5)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=1e-4, weight_decay=3e-4)
    predictor.evaluate(test_corpus=test_corpus)
    # model.save('models/double_preceptron.model')
    '''

    """
    Training: 4079/4244 (96.11215834118755%)
    Validation: 359/472 (76.0593220338983%)
    Training (max): 3992/4244 (94.062205466541%)
    Validation (max): 366/472 (77.54237288135593%)
    ------------------------------
    Accuracy: 75.40500736377025 %
    """

    '''
    model = TriplePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=3)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=1e-3)
    predictor.evaluate(test_corpus=test_corpus)
    '''

    """
    Training (max): 3732/4244 (87.9359095193214%)
    Validation (max): 372/472 (78.8135593220339%)
    ------------------------------
    Accuracy: 77.46686303387335 %
    """

    '''
    model = TriplePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=20, nb_classes=3, drop_out=0.5)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=1e-4, weight_decay=3e-4)
    predictor.evaluate(test_corpus=test_corpus)
    '''


def test_model_3_interactive():

    # TODO - encode this in unit tests

    """
    > quantity was wrong
    fix
    > add new screen for collateral agreements
    feat
    > refactor screen of collateral agreements
    refactor
    > move CollateralAgreement to module collateral
    refactor
    > extract computeQuantity from Trade class
    refactor
    > improve performance of cash sweeping
    feat
    > improve performance of CashSweeping method sweepAllPastCash
    feat
    > memory corruption
    fix
    > use smart pointer to simplify memory management
    refactor
    > clean code in getQuantity
    refactor
    > delete useless method getQuantity
    refactor
    > empty capital structures are not saved anymore
    fix
    """

    model = DoublePerceptronModel.load('models/double_preceptron.model')
    training_corpus = CommitMessageCorpus.from_split('train')
    bi_gram_tokenizer = BiGramTokenizer(NltkTokenizer())
    vectorizer = CollapsedOneHotVectorizer.from_corpus(training_corpus, bi_gram_tokenizer, min_freq=2)

    predictor = Predictor(model=model, vectorizer=vectorizer)

    test_corpus = CommitMessageCorpus.from_split('test')
    predictor.evaluate(test_corpus=test_corpus)
    predictor.show_errors(test_corpus=test_corpus)

    while True:
        sentence = input("> ")
        print(predictor.predict(sentence))

    # TODO - implement a kind of exception mechanism for this... based on CL? based on matching the fix description?
    # TODO - load the exceptions in the CorpusLoader and replace while reading the file...

    """
    Example of miss classifications:
    
    {COLLAT}(Openness): Fix few sonar issues
    > Predicted fix
    > Actual refactor
    
    {COL_BAU}(Collateral): Replace Calendar by LocalDateTime for start date and end date
    > Predicted fix
    > Actual refactor
    
    {COLLAT_BAU}(SnapshotGeneration): enhance error message
    > Predicted feat
    > Actual fix
    
    {COL_BAU}(AgreementInfo): integrate agreementInfo into the static data cache
    > Predicted feat
    > Actual refactor
    
    {COLLAT}(Agreement config): manage errors in the agreement cache
    > Predicted fix
    > Actual feat
    
    {COL_BAU}(collat_algo_service target refactoring) Refactor to be able to have a mock dll to be used for other target unit tests.
    > Predicted refactor
    > Actual feat
    
    {COL_BAU}(Openness cleaning): Clean deprecated code after openess development. Remove all backward compatibility code (except migration code).
    > Predicted refactor
    > Actual fix
    
    {COL_BAU}(Openness cleaning): Clean deprecated code after openess development. Remove valuationContext classes.
    > Predicted refactor
    > Actual fix
    """

    # What the fuck fixes

    """
    [COLLATERAL](Fix): reference returned instead of copy
    """


# test_model_3(split_seed=0, with_bi_grams=True)
# test_model_3_interactive()
