import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import *


class MLPSorting(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, input_size))

    def forward(self, x):
        return self.fc(x)


class SortingPredictor:
    def __init__(self, model):
        self.model = model

    def fit(self, data_set, epoch: int, learning_rate: float, weight_decay: float):
        loss_fct = nn.L1Loss() # TODO - another one
        optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        training_loader = DataLoader(data_set, batch_size=100, shuffle=True)

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
        return self.model(sequence.unsqueeze(0)).squeeze(0)


def test_sorting(input_size):
    xs = []
    ys = []
    for _ in range(10000):
        x = np.random.randint(0, 1000, input_size)
        xs.append(x)
        ys.append(np.array(sorted(x), dtype=np.int64))

    xs = torch.FloatTensor(np.stack(xs))
    ys = torch.FloatTensor(np.stack(ys))
    data_set = TensorDataset(xs, ys)

    model = MLPSorting(input_size=input_size, hidden_size=40)
    predictor = SortingPredictor(model=model)
    predictor.fit(data_set, epoch=10, learning_rate=1e-2, weight_decay=0)
    print(predictor.predict(np.random.randint(0, 1000, input_size)))


test_sorting(input_size=15)
