from annoy import AnnoyIndex

import copy
import fastText
import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch.utils.data import DataLoader

from dl.SplitDataset import *
from dl.Classification import ClassificationPredictor

# https://fasttext.cc/docs/en/english-vectors.html


def clean_text():
    with open('WordEmbeddingsInput.txt', 'r') as f_in:
        with open('WordEmbeddingsOutput.txt', 'w') as f_out:
            for line in f_in:
                words = nltk.word_tokenize(line)
                words = [w.lower() for w in words]
                out_line = " ".join(words)
                f_out.write(out_line + "\n")


def promixity_queries():
    embedding_size = 20

    model = fastText.train_unsupervised(input='WordEmbeddingsOutput.txt', dim=embedding_size, model='skipgram')
    known_words = set(model.get_words())

    # To query for nearest neighbors
    index = AnnoyIndex(embedding_size)
    for word in model.get_words():
        index.add_item(model.get_word_id(word), model.get_word_vector(word))
    index.build(5)
    # index.save('annoy.ann')

    # index.load('annoy.ann')
    while True:
        query = input("query>").split(' ')

        # analogy query
        if len(query) == 3:
            a, b, c = query
            if a in known_words and b in known_words and c in known_words:
                a, b, c = [model.get_word_vector(w) for w in [a, b, c]]
                result = index.get_nns_by_vector(vector=c + (b - a), n=10)
                print([model.get_words()[word_i] for word_i in result])
            else:
                print('unknown words', {a, b, c} - known_words)

        # proximity query
        elif len(query) == 1:
            v = model.get_word_vector(query[0])
            result = index.get_nns_by_vector(vector=v, n=10)
            print([model.get_words()[word_i] for word_i in result])

        # context query
        else:
            context = [model.get_word_vector(w) for w in query]
            vector = sum(context) / len(context)
            result = index.get_nns_by_vector(vector=vector, n=10)
            print([model.get_words()[word_i] for word_i in result])


class EmbeddingClassifier(nn.Module):
    def __init__(self, pretrained_embeddings, output_size):
        super().__init__()
        self.vocabulary_size = len(pretrained_embeddings)
        self.embedding_size = len(pretrained_embeddings[0])
        self.embedding = nn.Embedding(num_embeddings=self.vocabulary_size,
                                      embedding_dim=self.embedding_size)
        self.embedding.weight.data = torch.FloatTensor(pretrained_embeddings)
        self.embedding.weight.requires_grad = False
        self.fc = nn.Linear(in_features=self.embedding_size, out_features=output_size)

    def forward(self, x):
        x = self.embedding(x)   # batch_size, sequence_length, embedding_size
        x = x.sum(dim=1)
        x = self.fc(x)
        return fn.softmax(x, dim=-1)

    def parameters(self, recurse=True):
        # Avoid having the parameters of the pretrained weights
        return self.fc.parameters(recurse)


def test_classifier():
    embeddings = fastText.train_unsupervised(input='WordEmbeddingsOutput.txt', dim=20, model='skipgram')
    # print(embeddings.get_input_matrix().shape)  # what the heck... (2002034, 20)
    embeddings = np.stack([
        embeddings.get_word_vector(w)
        for w in embeddings.get_words()
    ])

    model = EmbeddingClassifier(pretrained_embeddings=embeddings, output_size=2)

    xs = [[1, 2, 3], [3, 2, 1]]
    xs = np.stack(np.array(x, dtype=np.int64) for x in xs)
    ys = np.stack([0, 1])

    data_set = SplitDataset(xs, ys)
    predictor = ClassificationPredictor(model=model)
    predictor.fit(data_set=data_set,learning_rate=1e-3, epoch=10)


# clean_text()
# promixity_queries()
# test_classifier()
