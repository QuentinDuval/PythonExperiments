import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import *


class CosModule(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class MLPPermutation2(nn.Module):
    """
    Very different model: we want to model the permutations (and not the values) by turning the problem in a
    classification problem (rather than a regression problem).

    The problem is that the output size in O(n²)
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size * input_size))

    def forward(self, x, apply_softmax=True):
        batch_size, sequence_size = x.shape
        x = self.fc(x)
        x = x.view((batch_size, sequence_size, sequence_size))
        if apply_softmax:
            x = fn.softmax(x, dim=-1)
        return x


class MLPPermutation3(nn.Module):
    """
    Again, we complexify the model to be able to find "odd" numbers: we add a binary representation of numbers
    """
    def __init__(self, input_size, binary_size, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size * binary_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size * input_size))

    def forward(self, x, apply_softmax=True):
        batch_size, sequence_size, repr_size = x.squeeze(-1).shape  # TODO - why the last element is size 1???
        x = x.view((batch_size, -1))
        x = self.fc(x)
        x = x.view((batch_size, sequence_size, sequence_size))
        if apply_softmax:
            x = fn.softmax(x, dim=-1)
        return x


class ClassificationPredictor:
    def __init__(self, model):
        self.model = model

    def fit(self, data_set, epoch: int, learning_rate: float, weight_decay: float):
        loss_fct = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        training_loader = DataLoader(data_set, batch_size=500, shuffle=True)

        for epoch in range(epoch):
            training_loss = 0
            self.model.train()
            for x, target in training_loader:
                self.model.zero_grad()
                outputs = self.model(x, apply_softmax=False)
                loss = loss_fct(outputs, target)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            print("Training:", training_loss)

    def predict(self, sequence):
        sequence = torch.FloatTensor(sequence)
        result = self.model(sequence.unsqueeze(0)).squeeze(0)
        _, predicted = torch.max(result.data, dim=-1)
        return [sequence[x].item() for x in predicted]


def fit_classification(xs, ys, model, epoch, learning_rate, weight_decay):
    xs = torch.FloatTensor(np.stack(xs))
    ys = torch.LongTensor(np.stack(ys))
    data_set = TensorDataset(xs, ys)

    predictor = ClassificationPredictor(model=model)
    predictor.fit(data_set, epoch=epoch, learning_rate=learning_rate, weight_decay=weight_decay)

    test_x = np.random.randint(low=0, high=100, size=xs.shape[-1])
    test_y = predictor.predict(test_x)
    print(test_x)
    print(test_y)
    return predictor


def test_copying(input_size, model, epoch):
    xs = []
    ys = []
    for _ in range(10000):
        x = np.random.randint(low=0, high=100, size=input_size)
        xs.append(x)
        ys.append(np.array(range(len(x)), dtype=np.int64))
    fit_classification(xs, ys, model, epoch=epoch, learning_rate=1e-2, weight_decay=0)


def test_reversing(input_size, model, epoch):
    xs = []
    ys = []
    for _ in range(10000):
        x = np.random.randint(low=0, high=100, size=input_size)
        xs.append(x)
        ys.append(np.array(range(len(x))[::-1], dtype=np.int64))
    fit_classification(xs, ys, model, epoch=epoch, learning_rate=1e-2, weight_decay=0)


def reverse_evens(xs):
    ys = list(range(len(xs)))
    i = 0
    j = len(ys) - 1
    while i < j:
        while i < len(ys) and xs[ys[i]] % 2:
            i += 1
        while j >= 0 and xs[ys[j]] % 2:
            j -= 1
        if i < j:
            ys[i], ys[j] = ys[j], ys[i]
            i += 1
            j -= 1
    return ys


def binary_repr(x):
    r = np.zeros((5, 1), dtype=np.int64)
    for i in range(5):
        if 0x1 & (x >> i):
            r[i] = 1
    return r


def test_reversing_evens(input_size, model, epoch):
    xs = []
    ys = []
    for _ in range(10000):
        x = np.random.randint(low=0, high=100, size=input_size)
        xs.append(np.array([binary_repr(v) for v in x], dtype=np.int64))
        ys.append(reverse_evens(x))

    xs = torch.FloatTensor(np.stack(xs))
    ys = torch.LongTensor(np.stack(ys))
    data_set = TensorDataset(xs, ys)

    predictor = ClassificationPredictor(model=model)
    predictor.fit(data_set, epoch=epoch, learning_rate=1e-2, weight_decay=0)

    x = np.random.randint(low=0, high=100, size=input_size)
    x_encoded = np.array([binary_repr(v) for v in x], dtype=np.int64)
    x_encoded = torch.FloatTensor(x_encoded)

    result = predictor.model(x_encoded.unsqueeze(0)).squeeze(0)
    _, predicted = torch.max(result.data, dim=-1)
    y = [x[i].item() for i in predicted]
    print(x, y)


def test_sorting(input_size, model, epoch):
    xs = []
    ys = []
    for _ in range(10000):
        x = np.random.randint(low=0, high=100, size=input_size)
        xs.append(x)
        y = list(range(len(x)))
        y.sort(key=lambda i: x[i])
        ys.append(np.array(y, dtype=np.int64))
    fit_classification(xs, ys, model, epoch=epoch, learning_rate=1e-2, weight_decay=0)


# test_copying(input_size=5, model=MLPPermutation2(input_size=5, hidden_size=10), epoch=10)
# test_copying(input_size=5, model=RNNPermutation2(hidden_size=10), epoch=50)

# test_reversing(input_size=5, model=MLPPermutation2(input_size=5, hidden_size=10), epoch=10)
# test_reversing(input_size=5, model=RNNPermutation2(hidden_size=20), epoch=50)

# TODO - does not learn (detection of even numbers... change the encoding?)
# test_reversing_evens(input_size=5, model=MLPPermutation3(input_size=5, binary_size=5, hidden_size=50), epoch=10)
# test_reversing_evens(input_size=5, model=RNNPermutation2(hidden_size=10), epoch=100)

# TODO - does not learn fully
# test_sorting(input_size=5, model=MLPPermutation2(input_size=5, hidden_size=25), epoch=100)
# test_sorting(input_size=5, model=RNNPermutation2(hidden_size=25), epoch=50)
