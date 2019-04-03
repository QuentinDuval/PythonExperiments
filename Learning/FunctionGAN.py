import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

from Learning.Vectorizer import *
from Learning.Vocabulary import *


class FunctionGenerator(nn.Module):
    def __init__(self, seed_size, output_size, vocab_size):
        super().__init__()
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.fc = nn.Sequential(
            nn.Linear(seed_size, self.output_size * self.vocab_size)
        )

    def forward(self, x):
        batch_size, seed_size = x.shape
        x = self.fc(x)
        x = x.view((batch_size, self.output_size, self.vocab_size))
        x = torch.argmax(x, dim=-1)     # To output the indices of the vocabulary
        return x


class FunctionDiscriminator(nn.Module):
    def __init__(self, input_size, vocab_size, embedding_size):
        super().__init__()
        self.embed = nn.Embedding(embedding_dim=embedding_size, num_embeddings=vocab_size)
        self.fc = nn.Sequential(
            nn.Linear(input_size * embedding_size, 1)
        )

    def forward(self, x, apply_softmax=False):
        batch_size = x.shape[0]
        x = self.embed(x)
        x = x.view((batch_size, -1))
        x = self.fc(x)
        if apply_softmax:
            x = fn.sigmoid(x)
        return x


class WordVectorizer(Vectorizer):
    def __init__(self, vocabulary, max_word_len):
        self.vocabulary = vocabulary
        self.max_word_len = max_word_len

    def vectorize(self, word):
        x = torch.zeros(self.max_word_len, dtype=torch.int64)
        for i, c in enumerate(word):
            x[i] = self.vocabulary.word_lookup(c)
        x[len(word):] = self.vocabulary.word_lookup(self.vocabulary.PADDING)
        return x


def iterate_batches():
    # TODO - a way to retrieve real functions
    pass


def test_gan():
    seed_size = 4
    batch_size = 100
    function_name_len = 20
    embedding_size = 10

    vocab = Vocabulary.from_words("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890", add_unknowns=True)
    vectorizer = WordVectorizer(vocabulary=vocab, max_word_len=function_name_len)

    generator = FunctionGenerator(seed_size=seed_size, output_size=function_name_len, vocab_size=len(vocab))
    discriminator = FunctionDiscriminator(input_size=function_name_len, vocab_size=len(vocab), embedding_size=embedding_size)

    objective = nn.BCELoss()   # TODO - with or without sigmoid?
    gen_optimizer = optim.Adam(params=generator.parameters(), lr=1e-3)
    dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=1e-3)

    x = vectorizer.vectorize("getQuantity")
    y = discriminator(x.unsqueeze(dim=0))
    print(y)

    seed = torch.zeros(seed_size, dtype=torch.float32)
    x = generator(seed.unsqueeze(dim=0))
    y = discriminator(x)
    print(y)


test_gan()




