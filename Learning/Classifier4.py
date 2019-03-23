import torch.nn.functional as fn

from Learning.Predictor import *
from Learning.WordEmbeddings import *

import numpy as np


class EmbeddingVectorizer(Vectorizer):
    """
    Vectorize a sentence by transforming each sentence to a vector filled with indices for each word
    """
    def __init__(self, vocabulary: Vocabulary, tokenizer, max_length):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_length = max_length

    def vectorize(self, sentence):
        features = np.zeros(self.max_length, dtype=np.int64)
        tokens = [self.vocabulary.START] + self.tokenizer(sentence) + [self.vocabulary.END]
        end_tokens = min(len(tokens), self.max_length)
        for i in range(end_tokens):
            features[i] = self.vocabulary.word_lookup(tokens[i])
        for i in range(end_tokens, self.max_length):
            features[i] = self.vocabulary.word_lookup(self.vocabulary.PADDING)
        return features

    def get_vocabulary_len(self):
        return len(self.vocabulary)

    @classmethod
    def from_corpus(cls, corpus: CommitMessageCorpus, tokenizer, max_length):
        vocabulary = Vocabulary.from_corpus(corpus=corpus, tokenizer=tokenizer, min_freq=2, add_unknowns=True)
        return cls(vocabulary=vocabulary, tokenizer=tokenizer, max_length=max_length)


class EmbeddingPerceptron(nn.Module):
    # TODO - test one with a simple sum on the embeddings!

    def __init__(self, vocabulary_len, sentence_size, embedding_size, nb_classes):
        super().__init__()
        self.embedding_size = embedding_size
        self.sentence_size = sentence_size
        self.nb_classes = nb_classes
        self.embed = nn.Embedding(vocabulary_len, self.embedding_size)
        self.output_layer = nn.Linear(self.embedding_size * self.sentence_size, self.nb_classes)

        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x, apply_softmax=True):
        batch_size = x.size()[0]
        x = self.embed(x)
        x = x.view((batch_size, -1))
        x = self.output_layer(x)
        if apply_softmax:
            x = fn.softmax(x, dim=-1)
        return x


def test_model_4(split_seed=None):
    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')

    vectorizer = EmbeddingVectorizer.from_corpus(training_corpus, NltkTokenizer(), max_length=30)
    vocab_len = vectorizer.get_vocabulary_len()
    sentence_len = vectorizer.max_length

    model = EmbeddingPerceptron(vocabulary_len=vocab_len, sentence_size=sentence_len, embedding_size=30, nb_classes=3)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=0.0001, weight_decay=0.0001)
    predictor.evaluate(test_corpus=test_corpus)

    # TODO - pre-trained embeddings do not bring much... (maybe accelerate convergence?)
    model = EmbeddingPerceptron(vocabulary_len=vocab_len, sentence_size=sentence_len, embedding_size=20, nb_classes=3)
    pretrained_embeddings = WordEmbeddings.load_from(model_path='resources/unsupervised_model.bin')
    pretrained_embeddings.transfer_learning_to(vectorizer.vocabulary, model.embed)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=0.0001, weight_decay=0.0001)
    predictor.evaluate(test_corpus=test_corpus)

    # TODO - compute the number of dimensions we get from this...
    # TODO - compare with what we get with classifier 3... just an embedding followed by a SUM (and this show in theory do great and be FAST)


"""
--------------------------------------------------
Training (max): 3669/4244 (86.45146088595665%)
Validation (max): 368/472 (77.96610169491525%)
------------------------------
Accuracy: 73.78497790868926 %

--------------------------------------------------
Training (max): 3428/4244 (80.77285579641847%)
Validation (max): 344/472 (72.88135593220339%)
------------------------------
Accuracy: 69.66126656848306 %
"""

# test_model_4(split_seed=0)
