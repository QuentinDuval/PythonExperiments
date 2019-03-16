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

    model = PerceptronModel(vocabulary_len=vocab_len, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus)
    predictor.evaluate(test_corpus=test_corpus)

    print("-" * 50)

    model = MultiPerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus)
    predictor.evaluate(test_corpus=test_corpus)

    print("-" * 50)

    model = TriplePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus)
    predictor.evaluate(test_corpus=test_corpus)

    print("-" * 50)

    model = TriplePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=0.0001, weight_decay=0.0001)
    predictor.evaluate(test_corpus=test_corpus)

    # TODO - try with dropout?


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


"""
Observe that:
1) by multiplying the number of layers we gain a bit of precision
2) the model does not generalizes as well (many dimensions) but weight_decay helps 

Model monolayer:
--------------------------------------------------
Training (max): 3753/4244 (88.43072573044299%)
Validation (max): 374/472 (79.23728813559322%)
------------------------------
Accuracy: 74.96318114874816 %

Model multi-layer
--------------------------------------------------
Training (max): 3709/4244 (87.39396795475966%)
Validation (max): 357/472 (75.63559322033898%)
------------------------------
Accuracy: 74.8159057437408 %

Model triple-layer
------------------------------
Training (max): 4056/4244 (95.57021677662583%)
Validation (max): 359/472 (76.0593220338983%)
------------------------------
Accuracy: 72.01767304860088 %

Model triple-layer (weight_decay = 0.0001)
--------------------------------------------------
Training (max): 3684/4244 (86.80490103675777%)
Validation (max): 380/472 (80.50847457627118%)
------------------------------
Accuracy: 74.96318114874816 %
"""

# test_model_3(split_seed=0)
# test_model_3_interactive()
