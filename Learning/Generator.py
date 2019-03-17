from Learning.Corpus import *
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


class CommitGenerationModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, hidden_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embed = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=embedding_size)
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, self.vocabulary_size)

    def forward(self, x):
        x = self.embed(x)           # Shape is batch_size, sequence_len, embedding_size
        x = x.permute(1, 0, 2)      # Shape is sequence_len, batch_size, embedding_size
        outputs, _ = self.rnn(x)

        sequence_len, batch_size, feat_size = outputs.shape
        outputs = outputs.contiguous().view(batch_size * sequence_len, feat_size)

        outputs = self.fc(outputs)              # Shape is batch_size * sequence_len, vocabulary_size
        outputs = fn.softmax(outputs, dim=-1)   # Makes sure that the probability sum to 1 at dimension -1
        return outputs.view(batch_size, sequence_len, self.vocabulary_size)


class CommitGenerationVectorizer(Vectorizer):
    def __init__(self, vocabulary: Vocabulary, tokenizer: Tokenizer, max_length=None):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_length = max_length

    def vectorize(self, sentence):
        tokens = [Vocabulary.START] + self.tokenizer(sentence) + [Vocabulary.END]
        tokens = tokens[:self.max_length] + (self.max_length - len(tokens)) * [Vocabulary.PADDING]
        tokens = [self.vocabulary.word_lookup(token) for token in tokens]
        return np.array(tokens, dtype=np.int64)


class CommitGenerationDataSet(Dataset):
    def __init__(self, prev, curr):
        self.prev = prev
        self.curr = curr

    def __len__(self):
        return len(self.prev)

    def __getitem__(self, index):
        return {'x': self.prev[index],
                'y': self.curr[index]}

    def split(self, ratio, seed=None):
        lhs, rhs = join_split(self.prev, self.curr, ratio, seed=seed)
        return CommitGenerationDataSet(*lhs), CommitGenerationDataSet(*rhs)

    @classmethod
    def from_corpus(cls, corpus: CommitMessageCorpus, vectorizer: CommitGenerationVectorizer):
        prev = []
        curr = []
        for line in itertools.chain(corpus.get_inputs(), corpus.get_unclassified()):
            tokens = vectorizer.vectorize(line)
            prev.append(np.array(tokens[:-1], dtype=np.int64))
            curr.append(np.array(tokens[1:], dtype=np.int64))
        return cls(prev, curr)


class GeneratorLossFunction:
    def __init__(self, vocabulary: Vocabulary):
        ignore_index = vocabulary.word_lookup(vocabulary.PADDING)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index) # Works better without the ignore index...
        self.loss_function = nn.CrossEntropyLoss()

    def __call__(self, outputs, labels):
        batch_size, sequence_len, embedding_size = outputs.shape
        outputs = outputs.view(batch_size * sequence_len, embedding_size)
        labels = labels.view(batch_size * sequence_len)
        return self.loss_function(outputs, labels)


def test_generator():
    # TODO - do on the full corpus
    training_corpus = CommitMessageCorpus.from_file('resources/manual_cl_list.txt', keep_unclassified=True)
    # training_corpus = CommitMessageCorpus.from_file('resources/perforce_cl_unsupervised.txt', keep_unclassified=True)

    tokenizer = NltkTokenizer()
    vocabulary = Vocabulary.from_corpus(corpus=training_corpus, tokenizer=tokenizer, min_freq=1, add_unknowns=True)
    vectorizer = CommitGenerationVectorizer(vocabulary=vocabulary, tokenizer=tokenizer, max_length=50)
    data_set = CommitGenerationDataSet.from_corpus(corpus=training_corpus, vectorizer=vectorizer)

    model = CommitGenerationModel(len(vocabulary), embedding_size=20, hidden_size=15)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=0)
    predictor.batch_size = 100
    predictor.loss_function = GeneratorLossFunction(vocabulary)
    predictor.fit_dataset(data_set=data_set, learning_rate=1e-3, weight_decay=0.0)

    res = predictor.predict("")
    print(res)


test_generator()


