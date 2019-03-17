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
    def __init__(self, with_bi_grams, split_seed=None):
        self.split_seed = split_seed
        self.training_corpus = CommitMessageCorpus.from_split('train')
        self.test_corpus = CommitMessageCorpus.from_split('test')

        self.vectorizer = CollapsedOneHotVectorizer.from_corpus(self.training_corpus, NltkTokenizer(), min_freq=2)
        self.vocab_len = self.vectorizer.get_vocabulary_len()

        if with_bi_grams:
            self.vectorizer = CollapsedOneHotVectorizer.from_corpus(
                self.training_corpus, tokenizer=BiGramTokenizer(NltkTokenizer()), min_freq=2)
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

        best_accuracy = 0.81
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


def test_model_3(with_bi_grams=False, split_seed=None):
    tester = Classifier3Test(with_bi_grams=with_bi_grams, split_seed=split_seed)

    print("single layer")
    # tester.test_single_layer()

    """
    Training (max): 3718/4221 (88.0833925610045%)
    Validation (max): 365/470 (77.6595744680851%)
    ------------------------------
    Accuracy: 77.11738484398218 %
    
    !!! BI GRAMS !!!
    
    Training (max): 4013/4221 (95.07225775882492%)
    Validation (max): 379/470 (80.63829787234043%)
    ------------------------------
    Accuracy: 77.4145616641902 %
    """

    print("double layer")
    tester.test_double_layers(rounds=1)

    """
    Training (max): 4015/4221 (95.1196398957593%)
    Validation (max): 355/470 (75.53191489361703%)
    ------------------------------
    Accuracy: 76.67161961367015 %
    
    Training (max): 3621/4221 (85.78535891968728%)
    Validation (max): 364/470 (77.4468085106383%)
    ------------------------------
    Accuracy: 76.96879643387817 %
    
    !!! BI GRAMS !!!
    
    Training (max): 3967/4221 (93.98246860933428%)
    Validation (max): 391/470 (83.19148936170212%)
    ------------------------------
    Accuracy: 76.82020802377416 %

    Training (max): 4052/4221 (95.99620942904525%)
    Validation (max): 390/470 (82.97872340425532%)
    --------------------------------------------------
    Accuracy: 78.00891530460625 %
    """

    print("triple layer")
    # tester.test_triple_layers()

    """
    Training (max): 3985/4221 (94.40890784174366%)
    Validation (max): 366/470 (77.87234042553192%)
    ------------------------------
    Accuracy: 75.33432392273403 %
    
    Training (max): 3667/4221 (86.87514806917792%)
    Validation (max): 367/470 (78.08510638297872%)
    ------------------------------
    Accuracy: 77.4145616641902 %
    
    !!! BI GRAMS !!!
    
    Training (max): 4057/4221 (96.11466477138119%)
    Validation (max): 356/470 (75.74468085106383%)
    ------------------------------
    Accuracy: 75.78008915304606 %

    Training (max): 3898/4221 (92.34778488509832%)
    Validation (max): 364/470 (77.4468085106383%)
    ------------------------------
    Accuracy: 76.67161961367015 %
    """


def test_model_3_interactive():

    model = DoublePerceptronModel.load('models/double_preceptron.model')
    training_corpus = CommitMessageCorpus.from_split('train')
    bi_gram_tokenizer = BiGramTokenizer(NltkTokenizer())
    vectorizer = CollapsedOneHotVectorizer.from_corpus(training_corpus, bi_gram_tokenizer, min_freq=2)

    predictor = Predictor(model=model, vectorizer=vectorizer)

    test_corpus = CommitMessageCorpus.from_split('test')
    predictor.evaluate(test_corpus=test_corpus)
    show_errors(predictor=predictor, test_corpus=test_corpus)

    print(verify_predictor(predictor))

    while True:
        sentence = input("> ")
        print(predictor.predict(sentence))


# test_model_3(split_seed=0, with_bi_grams=True)
# test_model_3_interactive()
