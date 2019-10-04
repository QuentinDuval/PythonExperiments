import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import *


# TODO:
#  For sorting numbers, no permutation seems to be used
#  https://datascience.stackexchange.com/questions/29345/how-to-sort-numbers-using-convolutional-neural-network

# TODO:
#  For even and odd numbers, this is hard
#  https://stats.stackexchange.com/questions/161189/train-a-neural-network-to-distinguish-between-even-and-odd-numbers


class MLPPermutation1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size))

    def forward(self, x, **kwargs):
        x = self.fc(x)
        # x = x.round() # Does not work, cannot be derived
        return x


class RNNPermutation1(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=1, hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x, **kwargs):
        batch_size, sequence_len = x.shape
        x = x.unsqueeze(dim=2)
        x = x.permute(1, 0, 2)
        init_state = torch.zeros(1, batch_size, self.hidden_size)
        x, final_state = self.rnn(x, init_state)
        x = x.permute(1, 0, 2)  # batch_size, sequence_len, self.hidden_size
        x = x.contiguous().view((batch_size * sequence_len, self.hidden_size))
        x = self.fc(x)
        x = x.view((batch_size, sequence_len))
        # x = x.round() # Does not work (cannot be derived)
        return x


class RegressionPredictor:
    def __init__(self, model):
        self.model = model

    def fit(self, data_set, epoch: int, learning_rate: float, weight_decay: float):
        loss_fct = nn.MSELoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        training_loader = DataLoader(data_set, batch_size=500, shuffle=True)

        for epoch in range(epoch):
            training_loss = 0
            self.model.train()
            for x, target in training_loader:
                self.model.zero_grad()
                outputs = self.model(x)
                loss = loss_fct(outputs, target)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            print("Training:", training_loss)

    def predict(self, sequence):
        sequence = torch.FloatTensor(sequence)
        result = self.model(sequence.unsqueeze(0)).squeeze(0)
        return [x.round().item() for x in result]


def fit_regression(xs, ys, model, epoch, learning_rate, weight_decay):
    xs = torch.FloatTensor(np.stack(xs))
    ys = torch.FloatTensor(np.stack(ys))
    data_set = TensorDataset(xs, ys)

    predictor = RegressionPredictor(model=model)
    predictor.fit(data_set, epoch=epoch, learning_rate=learning_rate, weight_decay=weight_decay)

    test_x = np.random.randint(low=0, high=100, size=xs.shape[-1])
    test_y = predictor.predict(test_x)
    print(test_x)
    print(test_y)


def test_copying(input_size, model, epoch):
    xs = []
    ys = []
    for _ in range(10000):
        x = np.random.randint(low=0, high=100, size=input_size)
        xs.append(x)
        ys.append(x)
    fit_regression(xs, ys, model, epoch=epoch, learning_rate=1e-2, weight_decay=0)


def test_reversing(input_size, model, epoch):
    xs = []
    ys = []
    for _ in range(10000):
        x = np.random.randint(low=0, high=100, size=input_size)
        xs.append(x)
        ys.append(x[::-1])
    fit_regression(xs, ys, model, epoch=epoch, learning_rate=1e-2, weight_decay=0)


def reverse_evens(xs):
    ys = list(xs)
    i = 0
    j = len(ys) - 1
    while i < j:
        while i < len(ys) and ys[i] % 2:
            i += 1
        while j >= 0 and ys[j] % 2:
            j -= 1
        if i < j:
            ys[i], ys[j] = ys[j], ys[i]
            i += 1
            j -= 1
    return ys


def test_reversing_evens(input_size, model, epoch):
    xs = []
    ys = []
    for _ in range(10000):
        x = np.random.randint(low=0, high=100, size=input_size)
        xs.append(x)
        ys.append(reverse_evens(x))
    fit_regression(xs, ys, model, epoch=epoch, learning_rate=1e-2, weight_decay=0)


def test_sorting(input_size, model, epoch):
    xs = []
    ys = []
    for _ in range(10000):
        x = np.random.randint(low=0, high=100, size=input_size)
        xs.append(x)
        ys.append(np.array(sorted(x), dtype=np.int64))
    fit_regression(xs, ys, model, epoch=epoch, learning_rate=1e-2, weight_decay=0)


# test_copying(input_size=5, model=MLPPermutation1(input_size=5, hidden_size=10), epoch=50)
# test_copying(input_size=5, model=RNNPermutation1(hidden_size=10), epoch=50)

# test_reversing(input_size=5, model=MLPPermutation1(input_size=5, hidden_size=10), epoch=50)
# test_reversing(input_size=5, model=RNNPermutation1(hidden_size=20), epoch=50)

# test_reversing_evens(input_size=5, model=MLPPermutation1(input_size=5, hidden_size=10), epoch=50)
# test_reversing_evens(input_size=5, model=RNNPermutation1(hidden_size=10), epoch=100)

# test_sorting(input_size=5, model=MLPPermutation1(input_size=5, hidden_size=50), epoch=50)
# test_sorting(input_size=5, model=RNNPermutation1(hidden_size=25), epoch=50)

