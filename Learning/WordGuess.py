import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.utils.data

from Learning.Ratio import *
from Learning.Tokenizer import *
from Learning.Vocabulary import *


"""
Deep learning model
"""


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
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
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


class RNNLanguageModeling(nn.Module):
    """
    Model that can be used for sequence modelling only
    """

    # TODO - produces tokens <unknown> which is a bug

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim=128):
        super().__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=False)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, apply_softmax=True):
        x = x.long()
        x = self.embeddings(x)          # Shape is batch_size, sequence_len, embedding_size
        x = x.permute(1, 0, 2)          # Shape is sequence_len, batch_size, embedding_size
        _, state = self.rnn(x)
        state = state.squeeze(dim=0)    # Squeeze the dimension for number of RNN layers
        x = self.fc(state)
        if apply_softmax:
            x = fn.softmax(x, dim=-1)
        return x


"""
Different ways to model the language mostly come from the way we build the data sets
"""


class DataSetFactory:
    def get_data_set(self, commits, tokenizer, vocabulary, window_size):
        pass


class LanguageModelingDataSet(DataSetFactory):
    def get_data_set(self, commits, tokenizer, vocabulary, window_size):
        xs = []
        ys = []
        context_size = window_size - 1
        padding = [vocabulary.PADDING] * context_size
        for commit in commits:
            tokens = tokenizer.tokenize(commit)
            tokens = padding + tokens + [vocabulary.END]
            for i in range(len(tokens) - context_size):
                context = tokens[i:i+context_size]
                next_word = tokens[i+context_size]
                context = np.array([vocabulary.word_lookup(w) for w in context], dtype=np.int64)
                next_word = vocabulary.word_lookup(next_word)
                xs.append(context)
                ys.append(next_word)

        xs = torch.LongTensor(xs)
        ys = torch.LongTensor(ys)
        return torch.utils.data.TensorDataset(xs, ys)


class CBOWModelingDataSet(DataSetFactory):
    def get_data_set(self, commits, tokenizer, vocabulary, window_size):
        xs = []
        ys = []
        middle = window_size // 2
        for commit in commits:
            tokens = tokenizer.tokenize(commit)
            for i in range(len(tokens) - window_size):
                context = tokens[i:i+middle] + tokens[i+middle+1:i+window_size]
                next_word = tokens[i+middle]
                context = np.array([vocabulary.word_lookup(w) for w in context], dtype=np.int64)
                next_word = vocabulary.word_lookup(next_word)
                xs.append(context)
                ys.append(next_word)

        xs = torch.LongTensor(xs)
        ys = torch.LongTensor(ys)
        return torch.utils.data.TensorDataset(xs, ys)


"""
Using the model to guess the next words from the context
"""


class WordGuesser:
    def __init__(self, model: nn.Module, vocabulary: Vocabulary):
        self.model = model
        self.vocabulary = vocabulary
        self.context_size = self.model.context_size

    def save(self, file_name):
        dump = {
            'vocabulary': self.vocabulary,
            'model': self.model
        }
        torch.save(dump, file_name)

    @classmethod
    def load(cls, file_name):
        dump = torch.load(file_name)
        return cls(model=dump['model'], vocabulary=dump['vocabulary'])

    def generate_sentence(self, max_words):
        padding_idx = self.vocabulary.word_lookup(self.vocabulary.PADDING)
        end_idx = self.vocabulary.word_lookup(self.vocabulary.END)
        tokens = [padding_idx] * self.context_size
        while True:
            if len(tokens) >= max_words:
                break
            context = torch.LongTensor([tokens[-self.context_size:]])
            token = self.generate_next(context)
            if token == end_idx:
                return self._to_text(tokens) + "."
            else:
                tokens.append(token)
        return self._to_text(tokens)

    def generate_next(self, context):
        probs = self.model(context)
        return torch.multinomial(probs, num_samples=1).item()

    def _to_text(self, tokens):
        parser = TokenParser()
        return " ".join(parser.generate(self.vocabulary.index_lookup(token)) for token in tokens[self.context_size:])


"""
Testing...
"""


def get_accuracy(outputs, targets):
    predicted = torch.argmax(outputs.data, dim=-1)
    return Ratio((predicted == targets).sum().item(), len(targets))


def train_word_predictor(window_size, data_set_factory: DataSetFactory, epoch_nb=20):
    corpus = CommitMessageCorpus.from_file('resources/perforce_cl_unsupervised.txt', keep_unclassified=True)
    tokenizer = NltkTokenizer()
    vocabulary = Vocabulary.from_corpus(corpus=corpus, tokenizer=tokenizer, min_freq=5, add_unknowns=True)
    vocab_size = len(vocabulary)
    print("Vocabulary size:", vocab_size)

    # commits = itertools.chain(corpus.get_inputs(), corpus.get_unclassified()) # TODO - use this for real data
    commits = corpus.get_inputs()
    data_set = data_set_factory.get_data_set(commits=commits,
                                             tokenizer=tokenizer,
                                             vocabulary=vocabulary,
                                             window_size=window_size)
    print("Data set size:", len(data_set))

    # model = WordPrediction(vocab_size=vocab_size, embedding_dim=20, context_size=window_size - 1, hidden_dim=128)
    model = RNNLanguageModeling(vocab_size=vocab_size, embedding_dim=20, context_size=window_size - 1, hidden_dim=128)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 0.98 ** e)

    for epoch in range(epoch_nb):
        total_loss = 0.0
        total_accuracy = Ratio(0, 0)
        for context, target in torch.utils.data.DataLoader(data_set, shuffle=True, batch_size=100): # If you augment the batch size, performance degrades...
            optimizer.zero_grad()
            output = model(context, apply_softmax=False)    # If you forget the =False, trains much less
            loss = objective(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            total_loss += loss.item()
            total_accuracy += get_accuracy(output, target)
        scheduler.step()

        print("-" * 20)
        print("Epoch:", epoch)
        print("Loss:", total_loss)
        print("Accuracy:", total_accuracy)

    return WordGuesser(model, vocabulary)


def train_commit_generation():
    guesser = train_word_predictor(window_size=4, data_set_factory=LanguageModelingDataSet(), epoch_nb=5)
    for _ in range(20):
        print(guesser.generate_sentence(max_words=50))
    guesser.save('models/generation.model')


def load_commit_generation():
    return WordGuesser.load('models/generation.model')


# train_commit_generation()
# generator = load_commit_generation()

# Works incredibly well (more than 47% at 20 iterations) => TODO - show what it guesses (show examples each iterations)
# train_word_predictor(window_size=5, data_set_factory=CBOWModelingDataSet())
