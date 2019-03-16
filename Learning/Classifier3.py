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
    Training (max): 3902/4244 (91.94156456173421%)
    Validation (max): 376/472 (79.66101694915254%)
    ------------------------------
    Accuracy: 75.69955817378498 %
    """

    '''
    model = PerceptronModel(vocabulary_len=vocab_len, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus)
    predictor.evaluate(test_corpus=test_corpus)
    '''

    """
    Training (max): 3803/4244 (89.60885956644675%)
    Validation (max): 373/472 (79.02542372881356%)
    ------------------------------
    Accuracy: 75.84683357879234 %
    """

    '''
    model = MultiPerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=1e-3, weight_decay=0)
    predictor.evaluate(test_corpus=test_corpus)
    '''

    """
    Training (max): 3647/4244 (85.93308199811499%)
    Validation (max): 380/472 (80.50847457627118%)
    ------------------------------
    Accuracy: 76.73048600883654 %
    """

    '''
    model = MultiPerceptronModel(vocabulary_len=vocab_len, hidden_dimension=25, nb_classes=4, drop_out=0.5)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=1e-3, weight_decay=3e-3)
    predictor.evaluate(test_corpus=test_corpus)
    '''

    """
    Training (max): 3904/4244 (91.98868991517436%)
    Validation (max): 373/472 (79.02542372881356%)
    ------------------------------
    Accuracy: 75.69955817378498 %
    """

    '''
    model = TriplePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=1e-4)
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
    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')

    vectorizer = CollapsedOneHotVectorizer.from_corpus(training_corpus, NltkTokenizer())
    vocab_len = vectorizer.get_vocabulary_len()

    model = TriplePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer)
    predictor.fit(training_corpus, learning_rate=0.001, weight_decay=0.0002)
    predictor.evaluate(test_corpus)

    while True:
        sentence = input("> ")
        print(predictor.predict(sentence))


# test_model_3(split_seed=0)
# test_model_3_interactive()
