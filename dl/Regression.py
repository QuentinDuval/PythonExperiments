import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(in_features=input_size, out_features=output_size, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x


class RegressionDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        return self.xs[item], self.ys[item]

    def split(self, ratio: float):
        indices = list(range(len(self)))
        split_point = int(ratio * len(indices))
        random.shuffle(indices)
        lhs, rhs = indices[:split_point], indices[split_point:]
        lhs_xs, rhs_xs = [self.xs[i] for i in lhs], [self.xs[i] for i in rhs]
        lhs_ys, rhs_ys = [self.ys[i] for i in lhs], [self.ys[i] for i in rhs]
        return RegressionDataset(lhs_xs, lhs_ys), RegressionDataset(rhs_xs, rhs_ys)


def fit_regression(model: nn.Module, data_set: RegressionDataset, epoch, learning_rate):
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


def predict_regression(model: nn.Module, number: float):
    input = torch.FloatTensor([[number]])
    output = model(input)
    print(output.item())


def test():
    model = LinearRegression(input_size=1, output_size=1)

    inputs = np.array([[np.random.uniform(-5, 5)] for _ in range(200)], dtype=np.float32)
    outputs = np.array([x+1 for x in inputs], dtype=np.float32)
    fit_regression(model, data_set=RegressionDataset(inputs, outputs), epoch=200, learning_rate=1e-1)

    print("Linear weights:", model.fc.weight.data)
    print("Linear bias:", model.fc.bias.data)

    while True:
        i = int(input("number"))
        predict_regression(model, i)


test()





