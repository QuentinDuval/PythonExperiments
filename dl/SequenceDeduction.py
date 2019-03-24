import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plot

from dl.SplitDataset import *


"""
Give a sequence to your algorithm, and it should deduce the next items
"""


class DeductionModel(nn.Module):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=1,
                          hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        state = np.zeros(shape=self.hidden_size, dtype=np.float32)
        _, final_state = self.rnn(x, state)
        x = final_state.squeeze(0)  # shape was: rnn_layers, batch_size, hidden_size
        x = self.fc(x)
        return x


def fit_regression(model: nn.Module, data_set: SplitDataset, epoch, learning_rate):
    loss_fct = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    training_set, validation_set = data_set.split(ratio=0.9)
    training_loader = DataLoader(training_set, batch_size=100, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=100, shuffle=False)

    for epoch in range(epoch):

        training_loss = 0
        model.train()
        for x, target in training_loader:
            outputs = model(x)
            loss = loss_fct(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        validation_loss = 0
        model.eval()
        for x, target in validation_loader:
            outputs = model(x)
            loss = loss_fct(outputs, target)
            validation_loss += loss.item()

        print("Training:", training_loss)
        print("Validation:", validation_loss)

    return model


def test_deduction():
    # TODO - how do you create the examples?
    #  * a language for pattern + generator?
    #  * give them the beginning of a function (sin?)
    # TODO - how do you make them of the same size for training...
    # TODO - what do you make them deduce?
    #  * the shape of the sin??? (2 parameters, period + phase?)
    #  * or the full shape of the curve?
    pass
