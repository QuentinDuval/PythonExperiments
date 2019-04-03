import copy
import fnmatch
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from typing import *

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
                                        # TODO - non differentiable... this is bad ! this is why it does not learn
                                        # https://becominghuman.ai/generative-adversarial-networks-for-text-generation-part-1-2b886c8cab10
                                        # https://arxiv.org/abs/1810.06640 (not easy...)
        return x


class FunctionDiscriminator(nn.Module):
    def __init__(self, input_size, vocab_size, embedding_size):
        super().__init__()
        self.embed = nn.Embedding(embedding_dim=embedding_size, num_embeddings=vocab_size)
        self.fc = nn.Sequential(
            nn.Linear(input_size * embedding_size, 1)
        )

    def forward(self, x, apply_softmax=True):
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
            if i >= self.max_word_len:
                break
            x[i] = self.vocabulary.word_lookup(c)
        x[len(word):] = self.vocabulary.word_lookup(self.vocabulary.PADDING)
        return x


def extract_functions_from_folder(absolute_path):
    # TODO there are problems: the pattern matches output types from typedef function pointers, class members, etc.
    pattern = re.compile("([\w]+)\(")
    with open("resources/function_names.txt", mode='a') as f_out:
        for root, dirnames, filenames in os.walk(absolute_path):
            for filename in fnmatch.filter(filenames, '*.h'):
                print(filename)
                with open(os.path.join(root, filename), mode='r') as f_in:
                    for line in f_in:
                        for match in pattern.findall(line):
                            f_out.writelines([match, "\n"])


def load_corpus():
    function_names = []
    with open("resources\\function_names.txt", mode="r") as f_in:
        for line in f_in:
            function_names.append(line.strip())
    return function_names


def corpus_to_data_set(corpus: List[str], vectorizer: Vectorizer):
    xs = []
    for function_name in corpus:
        x = vectorizer.vectorize(function_name)
        xs.append(x)
    return xs


class FunctionDataSet(Dataset):
    def __init__(self, inputs, target):
        self.xs = inputs
        self.ys = [target for _ in range(len(inputs))]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return {'x': self.xs[index],
                'y': self.ys[index]}


def test_gan():
    seed_size = 1
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

    function_names = load_corpus()
    real_data_set = FunctionDataSet(inputs=corpus_to_data_set(corpus=function_names, vectorizer=vectorizer), target=1)

    for epoch in range(10):

        """
        Discriminator trains to distinguish fakes from the generator from the real ones
        """

        generator.eval()
        discriminator.train()

        dis_data_set = copy.deepcopy(real_data_set)
        for _ in range(len(function_names) // batch_size):
            seed = torch.randn(batch_size, seed_size)
            for generated in generator(seed):
                dis_data_set.xs.append(generated.detach())
                dis_data_set.ys.append(0)

        dis_cumulative_loss = 0
        for minibatch in DataLoader(dis_data_set, batch_size=batch_size, shuffle=True):
            inputs, labels = minibatch['x'], minibatch['y']
            dis_optimizer.zero_grad()
            dis_outputs = discriminator(inputs).squeeze(dim=-1) # TODO - why the squeeze?
            dis_loss = objective(dis_outputs, labels.float())
            dis_loss.backward()
            dis_optimizer.step()
            dis_cumulative_loss += dis_loss.item()

        """
        Generator does its best to fool the discriminator
        """

        # TODO - does not learn - search for non differientiable things

        generator.train()
        discriminator.train()   # TODO - needed to propagate gradients?

        gen_cumulative_loss = 0
        for _ in range(len(function_names) // batch_size):
            generated = generator(seed)
            dis_outputs = discriminator(generated).squeeze(dim=-1) # TODO - why the squeeze?
            dis_loss = objective(dis_outputs, torch.ones(batch_size, dtype=torch.float32))
            dis_loss.backward()
            gen_optimizer.step()
            gen_cumulative_loss += dis_loss.item()

        print("Discriminator loss:", dis_cumulative_loss)
        print("Generator loss:", gen_cumulative_loss)

        generator.eval()
        seed = torch.randn(3, seed_size)
        for sample in generator(seed):
            s = ""
            for x in sample:
                word = vocab.index_lookup(x.item())
                if word not in vocab.RESERVED:
                    s += word
                else:
                    word += " "
            print(" -", s)



test_gan()
# extract_functions_from_folder("D:\\v3.1.build.dev.elm.bi.84806.sweeper\\lib\\bo\\spb\\h")



