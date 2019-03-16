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
    def from_corpus(cls, corpus: CommitMessageCorpus, tokenizer):
        vocabulary = Vocabulary.from_corpus(corpus=corpus, tokenizer=tokenizer, min_freq=2, add_unknowns=True)
        return cls(vocabulary=vocabulary, tokenizer=tokenizer)


def test_model_3(split_seed=None):
    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')

    vectorizer = CollapsedOneHotVectorizer.from_corpus(training_corpus, NltkTokenizer())
    vocab_len = vectorizer.get_vocabulary_len()

    """
    Training (max): 3779/4244 (89.04335532516494%)
    Validation (max): 371/472 (78.60169491525424%)
    ------------------------------
    Accuracy: 75.69955817378498 %
    """

    '''
    model = PerceptronModel(vocabulary_len=vocab_len, nb_classes=4)
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
    model = DoublePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=4)
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
    """

    '''
    model = DoublePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=40, nb_classes=4, drop_out=0.5)
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
    model = TriplePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=4)
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
    model = TriplePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=20, nb_classes=4, drop_out=0.5)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=1e-4, weight_decay=3e-4)
    predictor.evaluate(test_corpus=test_corpus)
    '''


def test_model_3_interactive():
    """
    > quantity was wrong
    fix
    > add new screen for collateral agreements
    feat
    > move CollateralAgreement to folder X
    refactor
    > extract computeQuantity from Contract class
    refactor
    """

    # TODO - bad results below

    """
    > memory corruption in MasterAgreement
    refactor
    > memory corruption
    refactor
    > corruption
    feat
    > memory fix
    fix
    > memory
    refactor
    """

    # TODO - save the vocabulary as well?

    model = DoublePerceptronModel.load('models/double_preceptron.model')
    training_corpus = CommitMessageCorpus.from_split('train')
    vectorizer = CollapsedOneHotVectorizer.from_corpus(training_corpus, NltkTokenizer())
    predictor = Predictor(model=model, vectorizer=vectorizer)

    while True:
        sentence = input("> ")
        print(predictor.predict(sentence))


# test_model_3(split_seed=0)
# test_model_3_interactive()
