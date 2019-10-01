import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *

import numpy as np


class PrincipalComponentExtraction(nn.Module):
    """
    Equivalent of PCA using an auto-encoder:
    - Compared to classic PCA, based on SVD, you can less simply ask for K dimension (with a diff to cut to find K)
    - Here we provide the hidden size we want (no way to find the optimal one)
    - TODO: But using regularization, you can select only the most important axis (kind of find the K for us as well?)
    - TODO: this learns the basic probability distribution of a manifold, allowing to do (for instance) de-noising
            by projecting on that space
    """

    def __init__(self, feature_size: int, output_size: int):
        super().__init__()
        self.encoder = nn.Linear(feature_size, output_size)
        self.decoder = nn.Linear(output_size, feature_size)

    def forward(self, xs):
        ys = self.encoder(xs)
        return self.decoder(ys)

    def fit(self, inputs, nb_epoch: int = 100, learning_rate: float = 1e-3, weight_decay: float = 0.):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        training_set = torch.stack([torch.from_numpy(x) for x in inputs])
        training_loader = DataLoader(training_set, batch_size=100, shuffle=True)
        for _ in range(nb_epoch):
            total_loss = 0.
            for xs in training_loader:
                optimizer.zero_grad()
                ys = self(xs)
                loss = criterion(ys, xs)           # Compute the error
                loss.backward()                    # Propagate the gradient
                optimizer.step()
                total_loss += loss.item()
            print(total_loss)

    def transform(self, xs):
        return self.encode(xs)

    def encode(self, xs):
        self.eval()
        with torch.no_grad():
            xs = torch.stack([torch.from_numpy(x) for x in xs])
            return self.encoder(xs).numpy()

    def decode(self, xs):
        self.eval()
        with torch.no_grad():
            xs = torch.stack([torch.from_numpy(x) for x in xs])
            return self.decoder(xs).numpy()


def test():
    def new_point(x, y):
        return np.array([x, y, 2 * x + y], dtype=np.float32)

    def generate_inputs(size=100):
        xs = np.random.uniform(low=-10, high=10, size=size)
        ys = np.random.normal(loc=1, scale=10, size=size)
        return [new_point(x, y) for x, y in zip(xs, ys)]

    inputs = generate_inputs(size=1000)
    pca = PrincipalComponentExtraction(feature_size=3, output_size=2)
    pca.fit(inputs=inputs, nb_epoch=300, learning_rate=1e-3, weight_decay=0.)

    tests = [new_point(1, 1), new_point(100, 100)]  # In and out of training distribution
    encoded = pca.transform(tests)
    decoded = pca.decode(encoded)
    for i in range(len(tests)):
        print(tests[i], "=>", decoded[i])


test()
