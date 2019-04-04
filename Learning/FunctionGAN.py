import copy
import fnmatch
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from Learning.Vectorizer import *
from Learning.Vocabulary import *


class FunctionGenerator(nn.Module):
    def __init__(self, seed_size, output_size, vocab_size, hidden_size):
        super().__init__()
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.fc = nn.Sequential(
            nn.Linear(seed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size * self.vocab_size)
        )

    def forward(self, x, with_argmax=True):
        batch_size, seed_size = x.shape
        x = self.fc(x)
        x = x.view((batch_size, self.output_size, self.vocab_size))

        """
        Beware, this function is not differentiable
        - Should only be used outside of training (to generate function names)
        - For training, output the differentiable full matrix (transformed into probabilities)
        Similar problems found in these resources:
        - https://becominghuman.ai/generative-adversarial-networks-for-text-generation-part-1-2b886c8cab10
        - https://arxiv.org/abs/1810.06640 (not easy...)
        """
        if with_argmax:
            x = torch.argmax(x, dim=-1)
        else:
            x = fn.softmax(x)
        return x


class FunctionDiscriminator(nn.Module):
    def __init__(self, input_size, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.fc = nn.Sequential(
            nn.Linear(input_size * embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, apply_softmax=True):
        """
        Two kinds of inputs are accepted here:
        - A batch of vector of indices (encoding of a function name)
        - A batch of matrices (for training with as a GAN, to allow back-propagation to the generator)
          In that case, the last dimension is the probability of each character to multiply with the embeddings:
          batch_size, function_name_size, probability_of_characters
        """

        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = self.embed(x)
        else:
            """
            We want to obtain a vector of shape (sentence_size, embedding_size) where the embedding_size is computed
            as the sum of the embedding weighted with the probabilities of each character.
            
            What we have:
            - the shape of the embedding matrix is (vocab_size, embedding_size)
            - we need to multiply it with a vector of shape (batch_size, sentence_size, vocab_size)
            
            What we need to do is:
            - reshape the inputs to be (batch_size * sentence_size, vocab_size)
            - do a batch product to obtain (batch_size * sentence_size, embedding_size)
            - reshape the outputs to (batch_size, sentence_size, embedding_size)
            
            Example:
            
                names = torch.FloatTensor(  # Batch of 2 names
                [
                    # First name (size 3 with 4 possible characters)
                    [[0.2, 0.5, 0.2, 0.1],
                     [0.2, 0.2, 0.5, 0.1],
                     [0.2, 0.1, 0.2, 0.5]],
            
                    # Second name (size 3 with 4 possible characters)
                    [[0.2, 0.5, 0.2, 0.1],
                     [0.2, 0.2, 0.5, 0.1],
                     [0.2, 0.1, 0.2, 0.5]]
                ])
                            
                embeddings = nn.Embedding(num_embeddings=4, embedding_dim=6)
                embed_matrix = embeddings.weight.data
                                
                names = names.view((2 * 3, 4))
                y = torch.matmul(names, embed_matrix)
                y = y.view((2, 3, -1))
                
                print(y)
            
            """
            batch_size, sentence_len, vocab_size = x.shape
            x = x.view((batch_size * sentence_len, vocab_size))
            x = torch.matmul(x, self.embed.weight.data)
            x = x.view((batch_size, sentence_len, -1))

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

    generator = FunctionGenerator(seed_size=seed_size, output_size=function_name_len,
                                  vocab_size=len(vocab), hidden_size=150)
    discriminator = FunctionDiscriminator(input_size=function_name_len, vocab_size=len(vocab),
                                          embedding_size=embedding_size, hidden_size=150)

    objective = nn.BCELoss()   # TODO - with or without sigmoid?
    gen_optimizer = optim.Adam(params=generator.parameters(), lr=1e-3)
    dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=1e-3)

    function_names = load_corpus()
    real_data_set = FunctionDataSet(inputs=corpus_to_data_set(corpus=function_names, vectorizer=vectorizer), target=1)

    for epoch in range(100):

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

        generator.train()
        discriminator.train()

        gen_cumulative_loss = 0
        for _ in range(len(function_names) // batch_size):
            generated = generator(seed, with_argmax=False) # Transmit a full matrix in order to learn
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

    return {'generator': generator,
            'discriminator': discriminator,
            'vectorizer': vectorizer}


def is_function(discriminator: FunctionDiscriminator, vectorizer: Vectorizer, function_name: str):
    x = vectorizer.vectorize(function_name)
    y = discriminator(x.unsqueeze(0))
    return y


"""
Running some tests
"""

# extract_functions_from_folder("D:\\v3.1.build.dev.elm.bi.84806.sweeper\\lib\\bo\\spb\\h")

# TODO - all of this is rather not working: the generator is not good enough, the discriminator is rather dumb at the end

result = test_gan()

discriminator = result['discriminator']
vectorizer = result['vectorizer']

while True:
    function_name = input("function_name>")
    print(is_function(discriminator, vectorizer, function_name))

