import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *

import numpy as np


# TODO - do the classic PCA and compute the complexity


class AutoEncoder:
    """
    Equivalent of PCA using an auto-encoder:
    - Compared to classic PCA, based on SVD, you can less simply ask for K dimension (with a diff to cut to find K)
    - Here we provide the hidden size we want (no way to find the optimal one)
    - TODO: But using regularization, you can select only the most important axis (kind of find the K for us as well?)
    - TODO: this learns the basic probability distribution of a manifold, allowing to do (for instance) de-noising
            by projecting on that space
    """

    def __init__(self, model):
        self.model = model

    def fit(self, inputs, nb_epoch: int = 100, learning_rate: float = 1e-3, weight_decay: float = 0.):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        training_set = torch.stack([torch.from_numpy(x) for x in inputs])
        training_loader = DataLoader(training_set, batch_size=100, shuffle=True)
        for _ in range(nb_epoch):
            total_loss = 0.
            for xs in training_loader:
                optimizer.zero_grad()
                ys = self.model(xs)
                loss = criterion(ys, xs)  # Compute the error
                loss.backward()  # Propagate the gradient
                optimizer.step()
                total_loss += loss.item()
            print(total_loss)

    def encode(self, xs):
        self.model.eval()
        with torch.no_grad():
            xs = torch.stack([torch.from_numpy(x) for x in xs])
            return self.model.encoder(xs).numpy()

    def decode(self, xs):
        self.model.eval()
        with torch.no_grad():
            xs = torch.stack([torch.from_numpy(x) for x in xs])
            return self.model.decoder(xs).numpy()


class LinearPCA(nn.Module):
    def __init__(self, feature_size: int, output_size: int):
        super().__init__()
        self.encoder = nn.Linear(feature_size, output_size)
        self.decoder = nn.Linear(output_size, feature_size)

    def forward(self, xs):
        ys = self.encoder(xs)
        return self.decoder(ys)


class TwoLayerPCA(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
        self.decoder = nn.Sequential(nn.Linear(output_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, feature_size))

    def forward(self, xs):
        ys = self.encoder(xs)
        return self.decoder(ys)


class ThreeLayerPCA(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_size, hidden_size),
                                     nn.ReLU(), nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(), nn.Linear(hidden_size, output_size))
        self.decoder = nn.Sequential(nn.Linear(output_size, hidden_size),
                                     nn.ReLU(), nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(), nn.Linear(hidden_size, feature_size))

    def forward(self, xs):
        ys = self.encoder(xs)
        return self.decoder(ys)


"""
Tests:
- linear approximation
- sphere approximation
"""


def test_linear():
    def new_point(x, y):
        return np.array([x, y, 2 * x + y], dtype=np.float32)

    def generate_inputs(size=100):
        xs = np.random.uniform(low=-10, high=10, size=size)
        ys = np.random.normal(loc=1, scale=10, size=size)
        return [new_point(x, y) for x, y in zip(xs, ys)]

    inputs = generate_inputs(size=1000)
    pca = AutoEncoder(model=LinearPCA(feature_size=3, output_size=2))
    pca.fit(inputs=inputs, nb_epoch=300, learning_rate=1e-3, weight_decay=0.)

    tests = [new_point(1, 1), new_point(100, 100)]  # In and out of training distribution
    encoded = pca.encode(tests)
    decoded = pca.decode(encoded)
    for i in range(len(tests)):
        print(tests[i], "=>", decoded[i])


def test_sphere():
    def new_point(alpha, beta):
        # Alpha is the angle from x to z, Beta is angle from x to y
        # Sum of x^2 + y^2 + z^2 is 1
        x = math.cos(alpha) * math.cos(beta)
        y = math.sin(beta)
        z = math.sin(alpha) * math.cos(beta)
        return np.array([x, y, z], dtype=np.float32)

    def generate_inputs(size=100):
        alphas = np.random.uniform(low=0, high=math.pi, size=size)
        betas = np.random.uniform(low=0, high=math.pi, size=size)
        return [new_point(a, b) for a, b in zip(alphas, betas)]

    inputs = generate_inputs(size=1000)
    pca = AutoEncoder(model=TwoLayerPCA(feature_size=3, hidden_size=20, output_size=2))
    # pca = AutoEncoder(model=ThreeLayerPCA(feature_size=3, hidden_size=20, output_size=2))
    pca.fit(inputs=inputs, nb_epoch=600, learning_rate=1e-3, weight_decay=1e-4)

    tests = [new_point(0, 0), new_point(1, 1)]
    encoded = pca.encode(tests)
    decoded = pca.decode(encoded)
    for i in range(len(tests)):
        print(tests[i], "=>", decoded[i])


def test_random():
    def new_point(x, y, z):
        return np.array([x, y, z], dtype=np.float32)

    def generate_inputs(size=100):
        xs = np.random.uniform(low=0, high=1, size=size)
        ys = np.random.uniform(low=0, high=1, size=size)
        zs = np.random.uniform(low=0, high=1, size=size)
        return [new_point(a, b, c) for a, b, c in zip(xs, ys, zs)]

    inputs = generate_inputs(size=1000)
    pca = AutoEncoder(model=TwoLayerPCA(feature_size=3, hidden_size=20, output_size=2))
    # pca = AutoEncoder(model=ThreeLayerPCA(feature_size=3, hidden_size=20, output_size=2))
    pca.fit(inputs=inputs, nb_epoch=600, learning_rate=1e-3, weight_decay=1e-4)

    tests = [new_point(0, 0, 0), new_point(0.5, 0.5, 0.5)]
    encoded = pca.encode(tests)
    decoded = pca.decode(encoded)
    for i in range(len(tests)):
        print(tests[i], "=>", decoded[i])


"""
Running tests
"""

# test_linear()
# test_sphere()
# test_random() # Cannot learn anything

# TODO - mention that it would not be able to learn anything if the compression was possible with multiple points
