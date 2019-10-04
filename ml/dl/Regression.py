import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plot


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(in_features=input_size, out_features=output_size, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x


class TwoLayerRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MultiLayerRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class MultiLayerPeriodicActivation(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = torch.cos(x)
        x = self.fc2(x)
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


def predict_regression(model: nn.Module, number: float):
    input = torch.FloatTensor([[number]])
    output = model(input)
    return output.item()


def test_linear(f):
    model = LinearRegression(input_size=1, output_size=1)

    inputs = np.array([[np.random.uniform(-5, 5)] for _ in range(200)], dtype=np.float32)
    outputs = np.array([f(x) for x in inputs], dtype=np.float32)
    fit_regression(model, data_set=SplitDataset(inputs, outputs), epoch=200, learning_rate=1e-1)

    print("Linear weights:", model.fc.weight.data)
    print("Linear bias:", model.fc.bias.data)

    xs = np.arange(-20.0, 20.0, .2)
    ys = np.array([predict_regression(model, x) for x in xs])
    real_ys = np.array([f(x) for x in xs])
    plot.plot(xs, ys, 'ro')
    plot.plot(xs, real_ys, 'b.')
    plot.show()


def test_feed_forward(f):
    # model = TwoLayerRegression(input_size=1, hidden_size=20, output_size=1)
    model = MultiLayerRegression(input_size=1, hidden_size=20, output_size=1)
    # model = MultiLayerPeriodicActivation(input_size=1, hidden_size=20, output_size=1)

    inputs = np.array([[np.random.uniform(-10, 10)] for _ in range(1000)], dtype=np.float32)
    outputs = np.array([f(x) for x in inputs], dtype=np.float32)
    fit_regression(model, data_set=SplitDataset(inputs, outputs), epoch=200, learning_rate=1e-1)

    xs = np.arange(-20.0, 20.0, .2)
    ys = np.array([predict_regression(model, x) for x in xs])
    real_ys = np.array([f(x) for x in xs])
    plot.plot(inputs, outputs, 'g.')
    plot.plot(xs, ys, 'ro')
    plot.plot(xs, real_ys, 'b.')
    plot.show()


"""
Polynomials are easy to approximate
"""

# test_linear(lambda x: x+1)
# test_feed_forward(lambda x: x**3 + 5 * x**2 + 10*x + 5 + np.random.uniform(-1, 1))
# test_feed_forward(lambda x: 20 + np.random.uniform(-10, 10))
# test_feed_forward(lambda x: x + np.random.uniform(-x, x))

# TODO - feed it with inputs from outside
# TODO - show that it is still easier to go for sklearn polynomial regression

"""
Periodic functions are much harder...
"""

# test_feed_forward(lambda x: math.cos(x))



