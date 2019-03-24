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


class OneHotAugmentation:
    """
    Attempt at augmenting the data set by combining 2 fixes / 2 refactors / 2 features
    Beware of not doing this before splitting the data set to avoid snooping
    """

    def __init__(self, ratio: float):
        self.ratio = ratio

    def __call__(self, data_set: CommitMessageDataSet):
        by_target = {}
        for x, y in zip(data_set.xs, data_set.ys):
            by_target.setdefault(y, []).append(x)
        for i in range(int(len(data_set) * self.ratio)):
            target_index = i % len(CommitMessageCorpus.TARGET_CLASSES)
            commits = by_target[target_index]
            a = random.choice(commits)
            b = random.choice(commits)
            data_set.xs.append(np.maximum(a, b))
            data_set.ys.append(target_index)


class Classifier3Test:
    def __init__(self, n_grams=1, split_seed=None):
        self.split_seed = split_seed
        self.training_corpus = CommitMessageCorpus.from_split('train')
        self.test_corpus = CommitMessageCorpus.from_split('test')

        self.vectorizer = CollapsedOneHotVectorizer.from_corpus(self.training_corpus, NltkTokenizer(), min_freq=2)
        self.vocab_len = self.vectorizer.get_vocabulary_len()

        if n_grams > 1:
            self.vectorizer = CollapsedOneHotVectorizer.from_corpus(
                self.training_corpus, tokenizer=NGramTokenizer(NltkTokenizer(), count=n_grams), min_freq=2)
            self.vocab_len = self.vectorizer.get_vocabulary_len()

    def predictor_for(self, model: nn.Module) -> Predictor:
        return Predictor(model, vectorizer=self.vectorizer, with_gradient_clipping=True, split_seed=self.split_seed)

    def train(self, predictor: Predictor, **kwargs):
        max_accuracy = predictor.fit(training_corpus=self.training_corpus, **kwargs)
        predictor.evaluate(test_corpus=self.test_corpus)
        return max_accuracy

    def test_single_layer(self):
        model = PerceptronModel(vocabulary_len=self.vocab_len, nb_classes=3)
        self.train(self.predictor_for(model))
        # model.save('models/preceptron.model')

    def test_double_layers(self, rounds=1):
        model = DoublePerceptronModel(vocabulary_len=self.vocab_len, hidden_dimension=100, nb_classes=3)
        self.train(self.predictor_for(model))

        best_accuracy = 0.84    # Best record so far
        for _ in range(rounds):
            model = DoublePerceptronModel(vocabulary_len=self.vocab_len, hidden_dimension=40, nb_classes=3, drop_out=0.5)
            accuracy = self.train(self.predictor_for(model), learning_rate=1e-4, weight_decay=3e-4)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                print("SAVING MODEL WITH ACCURACY", best_accuracy)
                model.save('models/double_preceptron.model')

    def test_triple_layers(self):
        model = TriplePerceptronModel(vocabulary_len=self.vocab_len, hidden_dimension=100, nb_classes=3)
        self.train(self.predictor_for(model))

        model = TriplePerceptronModel(vocabulary_len=self.vocab_len, hidden_dimension=20, nb_classes=3, drop_out=0.5)
        self.train(self.predictor_for(model), learning_rate=1e-4, weight_decay=3e-4)


def test_model_3(n_grams=1, split_seed=None):
    tester = Classifier3Test(n_grams=n_grams, split_seed=split_seed)

    print("single layer")
    tester.test_single_layer()

    print("double layer")
    tester.test_double_layers(rounds=10)

    print("triple layer")
    tester.test_triple_layers()


# test_model_3(split_seed=0, n_grams=3)
# test_model_3_interactive()
