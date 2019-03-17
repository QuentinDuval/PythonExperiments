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
    Training (max): 3952/4221 (93.62710258232646%)
    Validation (max): 385/470 (81.91489361702128%)
    ------------------------------
    Accuracy: 78.00891530460625 %
    """

    '''
    model = DoublePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=40, nb_classes=3, drop_out=0.5)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    # predictor.data_augmentation = OneHotAugmentation(ratio=3.0) # Does not bring anything here
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


test_model_3(split_seed=0, with_bi_grams=True)
# test_model_3_interactive()
