import numpy as np

from Learning.Classifier2 import *
from Learning.Predictor import *
from Learning.Tokenizer import *
from Learning.Vocabulary import *
from Learning.Verification import *


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


"""
Packaging the model
"""


class CommitClassifier3:
    def __init__(self, model, vocabulary: Vocabulary, n_grams=1):
        self.model = model
        self.vocabulary = vocabulary
        self.n_grams = n_grams
        if n_grams > 1:
            self.tokenizer = NGramTokenizer(NltkTokenizer(), count=n_grams)
        else:
            self.tokenizer = NltkTokenizer()
        self.vectorizer = CollapsedOneHotVectorizer(vocabulary=vocabulary, tokenizer=self.tokenizer)
        self.predictor = Predictor(model=model, vectorizer=self.vectorizer)

    def predict(self, sentence: str):
        return self.predictor.predict(sentence)

    def save(self, file_name):
        dump = {
            'model': self.model,
            'vocabulary': self.vocabulary,
            'n_grams': self.n_grams
        }
        torch.save(dump, file_name)

    @classmethod
    def load(cls, file_name):
        dump = torch.load(file_name)
        classifier = cls(model=dump['model'],
                         vocabulary=dump['vocabulary'],
                         n_grams=dump['n_grams'])
        return classifier


"""
Tests...
"""


class Classifier3Test:
    def __init__(self, n_grams=1, split_seed=None):
        self.n_grams = n_grams
        self.split_seed = split_seed
        self.training_corpus = CommitMessageCorpus.from_split('train')
        self.test_corpus = CommitMessageCorpus.from_split('test')
        if n_grams > 1:
            self.tokenizer = NGramTokenizer(NltkTokenizer(), count=n_grams)
        else:
            self.tokenizer = NltkTokenizer()
        self.vectorizer = CollapsedOneHotVectorizer.from_corpus(self.training_corpus, tokenizer=self.tokenizer, min_freq=2)
        self.vocabulary = self.vectorizer.vocabulary
        self.vocab_len = len(self.vocabulary)

    def train(self, predictor: Predictor, **kwargs):
        predictor.with_gradient_clipping = True
        # predictor.split_seed = self.split_seed
        max_accuracy = predictor.fit(training_corpus=self.training_corpus, **kwargs)
        predictor.evaluate(test_corpus=self.test_corpus)
        return max_accuracy

    def test_save(self, classifier: CommitClassifier3, file_name):
        for commit in ['quantity was wrong',
                       'move CollateralAgreement to lib/folder1/folder2',
                       'add tab in collateral screen to show statistics',
                       'use smart pointers to simplify memory management of ClassName']:
            print(commit)
            print(classifier.predict(commit))
        answer = input('save model (Y/N)? >')
        if answer.lower() == "y":
            print("saving model...")
            classifier.save(file_name)
            return True
        return False

    def test_single_layer(self):
        model = PerceptronModel(vocabulary_len=self.vocab_len, nb_classes=3)
        classifier = CommitClassifier3(model=model, vocabulary=self.vocabulary, n_grams=self.n_grams)
        self.train(classifier.predictor)
        self.test_save(classifier, 'models/single_preceptron.model')

    def test_double_layers(self, rounds=1):
        best_accuracy = 0.0
        for _ in range(rounds):
            model = DoublePerceptronModel(vocabulary_len=self.vocab_len, hidden_dimension=40, nb_classes=3, drop_out=0.5)
            classifier = CommitClassifier3(model=model, vocabulary=self.vocabulary, n_grams=self.n_grams)
            accuracy = self.train(classifier.predictor, learning_rate=1e-4, weight_decay=3e-4, learning_rate_decay=0.98)
            if accuracy >= best_accuracy and self.test_save(classifier, 'models/double_preceptron.model'):
                best_accuracy = accuracy

    def test_triple_layers(self):
        model = TriplePerceptronModel(vocabulary_len=self.vocab_len, hidden_dimension=20, nb_classes=3, drop_out=0.5)
        classifier = CommitClassifier3(model=model, vocabulary=self.vocabulary, n_grams=self.n_grams)
        self.train(classifier.predictor, learning_rate=1e-4, weight_decay=3e-4)
        self.test_save(classifier, 'models/triple_preceptron.model')


def test_model_3(n_grams=1, split_seed=None):
    tester = Classifier3Test(n_grams=n_grams, split_seed=split_seed)

    print("single layer")
    # tester.test_single_layer()

    print("double layer")
    # tester.test_double_layers(rounds=10)

    print("triple layer")
    tester.test_triple_layers()


def load_best_model():
    # classifier = CommitClassifier3.load('models/double_preceptron.model')
    classifier = CommitClassifier3.load('models/triple_preceptron.model')
    return classifier.predictor


def display_best_model_errors():
    predictor = load_best_model()
    show_errors(predictor=predictor, test_corpus=CommitMessageCorpus.from_split('train'))


# test_model_3(split_seed=0, n_grams=3)
# display_best_model_errors()
