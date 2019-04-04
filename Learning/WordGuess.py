import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

from Learning.Ratio import *
from Learning.Tokenizer import *
from Learning.Vocabulary import *


class WordPrediction(nn.Module):
    """
    Model that can be used both for:
    - Language modelling (next word finding)
    - Continuous bag of word embedding learning

    Reference:
    https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
    """

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim=128):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(context_size * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x, apply_softmax=True):
        batch_size, context_size = x.shape
        x = self.embeddings(x).view((batch_size, -1))
        x = self.fc(x)
        if apply_softmax:
            x = fn.softmax(x, dim=-1)
        return x


def language_modeler_data_set(commits, tokenizer, vocabulary, window_size):
    xs = []
    ys = []
    for commit in commits:
        tokens = tokenizer.tokenize(commit)
        for i in range(len(tokens) - window_size):
            context = tokens[i:i+window_size-1]
            next_word = tokens[i+window_size-1]
            context = np.array([vocabulary.word_lookup(w) for w in context], dtype=np.int64)
            next_word = vocabulary.word_lookup(next_word)
            xs.append(context)
            ys.append(next_word)

    xs = torch.LongTensor(xs)
    ys = torch.LongTensor(ys)
    return torch.utils.data.TensorDataset(xs, ys)


# TODO - common bag of word data_set
# TODO - use the language modeler to generate commits (you have to use padding to do this...)


"""
Testing...
"""


def get_accuracy(outputs, targets):
    predicted = torch.argmax(outputs.data, dim=-1)
    return Ratio((predicted == targets).sum().item(), len(targets))


def test_language_modeler(window_size):
    corpus = CommitMessageCorpus.from_file('resources/perforce_cl_unsupervised.txt', keep_unclassified=True)
    tokenizer = NltkTokenizer()
    vocabulary = Vocabulary.from_corpus(corpus=corpus, tokenizer=tokenizer, min_freq=5, add_unknowns=True)
    vocab_size = len(vocabulary)
    print("Vocabulary size:", vocab_size)

    # commits = itertools.chain(corpus.get_inputs(), corpus.get_unclassified()) # TODO - use this for real data
    commits = corpus.get_inputs()
    data_set = language_modeler_data_set(commits, tokenizer, vocabulary, window_size=window_size)
    print("Data set size:", len(data_set))

    model = WordPrediction(vocab_size=vocab_size, embedding_dim=20, context_size=window_size - 1, hidden_dim=128)
    objective = nn.CrossEntropyLoss()   # TODO - Probably more the distance between target vector and output vector ?
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    for epoch in range(20):
        total_loss = 0.0
        total_accuracy = Ratio(0, 0)
        for context, target in torch.utils.data.DataLoader(data_set, shuffle=True, batch_size=1000):
            optimizer.zero_grad()
            output = model(context, apply_softmax=False)    # If you forget the =False, trains much less
            loss = objective(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            total_loss += loss.item()
            total_accuracy += get_accuracy(output, target)
            
        print("-" * 20)
        print("Epoch:", epoch)
        print("Loss:", total_loss)
        print("Accuracy:", total_accuracy)


torch.set_num_threads(4)
test_language_modeler(window_size=5)
