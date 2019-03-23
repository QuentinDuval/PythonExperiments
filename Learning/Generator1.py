from Learning.Corpus import *
from Learning.Predictor import *
from Learning.Predictor import *
from Learning.Tokenizer import *
from Learning.Vocabulary import *
from Learning.Utils import *

import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.utils.data import Dataset


"""
Language modeler for the commits, as done in
https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

A bit better than a markov chain, but not capable of getting the full context (like LSTM) 
"""


class CommitLanguageModeler(nn.Module):
    def __init__(self, context_size, embedding_size, vocabulary_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.fc1 = nn.Linear(context_size * embedding_size, 128)
        self.fc2 = nn.Linear(128, vocabulary_size)

    def forward(self, x, apply_softmax=True):
        # TODO - try to condition it with the type of commit - but then you have much less data...
        x = x.long()
        batch_size, context_size = x.shape
        x = self.embeddings(x)
        x = x.view((batch_size, -1))
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.fc2(x)
        if apply_softmax:
            x = fn.softmax(x, dim=-1)
        return x


class CommitLanguageModelVectorizer(Vectorizer):
    def __init__(self, vocabulary: Vocabulary, tokenizer):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer

    def vectorize(self, sentence):
        tokens = self.tokenizer(sentence)
        return self.vectorize_tokens(tokens)

    def vectorize_tokens(self, tokens):
        return np.array([self.vocabulary.word_lookup(token) for token in tokens], dtype=np.int64)

    def get_vocabulary_len(self):
        return len(self.vocabulary)


def language_modeler_dataset(corpus: CommitMessageCorpus,
                             vectorizer: Vectorizer,
                             tokenizer: Tokenizer,
                             context_size: int) -> CommitMessageDataSet:
    prev = []
    curr = []
    for line in itertools.chain(corpus.get_inputs(), corpus.get_unclassified()):
        tokens = tokenizer(line)
        # TODO - add the end of sentence!
        for i in range(context_size, len(tokens)):
            context = tokens[i-context_size:i]
            vectorized_context = vectorizer.vectorize_tokens(context)
            vectorized_word = vectorizer.vectorize_tokens([tokens[i]])
            prev.append(vectorized_context)
            curr.append(vectorized_word[0])
    return CommitMessageDataSet(prev, curr)


def predict_next_1(predictor: Predictor, sentence: str):
    x = torch.LongTensor(predictor.vectorizer.vectorize(sentence=sentence))
    x = x.unsqueeze(dim=0)
    predictor.model.eval()
    y = predictor.model(x)
    predicted = torch.multinomial(y, num_samples=1) # TODO - concept of temperature
    return predictor.vectorizer.vocabulary.index_lookup(predicted.item())


def test_generator_1(context_size):
    training_corpus = CommitMessageCorpus.from_file('resources/perforce_cl_train.txt', keep_unclassified=True)
    # training_corpus = CommitMessageCorpus.from_file('resources/perforce_cl_unsupervised.txt', keep_unclassified=True)

    tokenizer = NltkTokenizer()
    vocabulary = Vocabulary.from_corpus(corpus=training_corpus, tokenizer=tokenizer, min_freq=1, add_unknowns=True)

    vectorizer = CommitLanguageModelVectorizer(vocabulary=vocabulary, tokenizer=tokenizer)
    data_set = language_modeler_dataset(corpus=training_corpus, vectorizer=vectorizer, tokenizer=tokenizer, context_size=context_size)

    model = CommitLanguageModeler(vocabulary_size=len(vocabulary), embedding_size=20, context_size=context_size)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=0)
    predictor.max_epoch = 10
    predictor.fit_dataset(data_set=data_set, learning_rate=1e-3, weight_decay=0.0)

    while True:
        start = input("sentence>")
        if len(vectorizer.vectorize(start)) == context_size:
            print(predict_next_1(predictor, start))


# test_generator_1(context_size=2)
